import argparse
import shutil
import time

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import os
import numpy as np
import random
import dataset.dataset as dataset
import densenet as dn
import dataset.joint_transforms as joint_transforms
from evaluation import evaluate

parser = argparse.ArgumentParser(description='Pytorch RemoteNet Training')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    help='mini-batch size(default:1)')
parser.add_argument('-lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--power', default=0.9, type=float,
                    help='lr power (default: 0.9)')
parser.add_argument('--print-freq', default=10, type=int,
                    help='print frequency(default: 10)')
parser.add_argument('--num-class', default=6, type=int,
                    help='number of class(default: 5)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation(default: True)')
parser.add_argument('--data-root', default='/home/jinqizhao/dataset/image/Remote_sensing/potsdam/2_Ortho_RGB_seg/',
                    type=str, help='path to data')
parser.add_argument('--label-root', default='/home/jinqizhao/dataset/image/Remote_sensing/potsdam/Label_gray_no_Boundary/',
                    type=str, help='path to label')
parser.add_argument('--label-list', default='./dataset/list/top_potsdam_noBoundary.txt', type=str,
                    help='label list')
parser.add_argument('--result-pth', default='./result/', type=str,
                    help='result path')
parser.add_argument('--resume', default='', type=str,
                    help='path to latset checkpoint(default: None')
parser.add_argument('--name', default='RemoteNet', type=str,
                    help='name of experiment')
parser.set_defaults(augment=True)

best_record = {'epoch': 0, 'val_loss': 0.0, 'acc': 0.0, 'miou': 0.0}


def main():
    global args, best_record
    args = parser.parse_args()

    if args.augment:
        transform_train = joint_transforms.Compose([
            joint_transforms.RandomCrop(384),
            joint_transforms.Scale(400),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomVerticallyFlip(),
            joint_transforms.Rotate(90),
            ])
    else:
        transform_train = None

    label_train, label_val = split_dataset(args.label_list)

    train_file = open('ISPRS_train.txt', 'w')
    for i in label_train:
        train_file.write(i+'\n')
    train_file.close()

    val_file = open('ISPRS_val.txt', 'w')
    for i in label_val:
        val_file.write(i+'\n')
    val_file.close()

    dataset_train = dataset.RSData('train', args.data_root, args.label_root, label_train, transform_train)
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)

    dataset_val = dataset.RSData('val', args.data_root, args.label_root, label_val)
    dataloader_val = data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=None, num_workers=8)

    model = dn.DenseNet(
        growth_rate=32, num_layers=[6, 12, 48, 32], theta=0.5,
        num_classes=args.num_class + 1, input_size=(400, 400), dropout_rate=0.2
    )

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
    ))

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # define loss function (criterion) and pptimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(dataloader_train, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc, mean_iou, val_loss = validate(dataloader_val, model, criterion, args.result_pth, epoch)

        is_best = mean_iou > best_record['miou']
        if is_best:
            best_record['epoch'] = epoch
            best_record['val_loss'] = val_loss.avg
            best_record['acc'] = acc
            best_record['miou'] = mean_iou
        save_checkpoint({
            'epoch': epoch + 1,
            'val_loss': val_loss.avg,
            'accuracy': acc,
            'miou': mean_iou,
            'state_dict': model.state_dict(),
        }, is_best)

        print('------------------------------------------------------------------------------------------------------')
        print('[epoch: %d], [val_loss: %5f], [acc: %.5f], [miou: %.5f]' % (
            epoch, val_loss.avg, acc, mean_iou))
        print('best record: [epoch: {epoch}], [val_loss: {val_loss:.5f}], [acc: {acc:.5f}], [miou: {miou:.5f}]'
              .format(**best_record))
        print('------------------------------------------------------------------------------------------------------')


def train(dataloader_train, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    weights = [0.25, 0.5, 0.75, 1]

    model.train()

    end = time.time()
    for i, (input_, target) in enumerate(dataloader_train):
        target = target.cuda(async=True)
        input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = 0
        j = 0
        for l in output:
            loss += (weights[j] * criterion(l, target_var))
            j += 1

        # record loss
        losses.update(loss.data[0], input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(dataloader_train),
                   batch_time=batch_time, loss=losses))


def validate(dataloader_val, model, criterion, result_pth, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    target_list = []
    pred_list = []

    model.eval()

    end = time.time()
    for i, (input_, target) in enumerate(dataloader_val):
        target = target.cuda(async=True)
        input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output[-1], target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input_.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        for j in range(target.shape[0]):
            target_list.append(target.cpu()[j].numpy())
            pred_list.append(np.argmax(output[-1].cpu().data[j].numpy(), axis=0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(dataloader_val), batch_time=batch_time, loss=losses))

    acc, mean_iou = evaluate(target_list, pred_list, args.num_class + 1, result_pth, epoch)
    return acc, mean_iou, losses


def adjust_learning_rate(optimizer, epoch):
    lr = args.learning_rate*((1-float(epoch)/args.epochs)**args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def split_dataset(label_list):
    id_v = random.sample(range(24), 10)
    id_t = range(24)
    for i in id_v:
        id_t.remove(i)
    label = [i_id.strip() for i_id in open(label_list)]
    train_label = []
    val_label = []
    for j in id_t:
        train_label += label[j*400:(j+1)*400]
    for k in id_v:
        val_label += label[k*400:(k+1)*400]
    return train_label, val_label


def save_checkpoint(state, is_best, filename='checkpoint.pth.rar'):
    """Saves checkpoint to disk"""
    directory = "runs_No_normalize/%s/" % args.name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs_No_normalize/%s/' % args.name + 'model_best.pth.tar')


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
