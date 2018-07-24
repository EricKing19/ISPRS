import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DenseBlock(nn.Sequential):
    def __init__(self, num_layer, num_features, growth_rate, dropout_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layer):
            layer = DenseLayer(num_features + i * growth_rate, growth_rate, dropout_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseLayer(nn.Sequential):
    def __init__(self, num_features, growth_rate, dropout_rate=0.0):
        super(DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_features))
        self.add_module('relu.1', nn.ReLU(inplace=True))
        self.add_module('conv.1', nn.Conv2d(num_features, 4 * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('drop.1', nn.Dropout(dropout_rate, inplace=False))
        self.add_module('norm.2', nn.BatchNorm2d(4 * growth_rate))
        self.add_module('relu.2', nn.ReLU(inplace=True))
        self.add_module('conv.2', nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('drop.2', nn.Dropout(dropout_rate, inplace=False))

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        return torch.cat([x, new_features], dim=1)


class TransitionLayer(nn.Module):
    def __init__(self, num_features, theta, stride=2, dropout_rate=0.0):
        super(TransitionLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.norm = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_features, int(num_features*theta), kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        if self.dropout_rate > 0:
            x = f.dropout(x, p=self.dropout_rate, inplace=False, training=self.training)
        x = self.pool(x)

        return x


class PSPModule(nn.Module):
    def __init__(self, num_features, size_series):
        super(PSPModule, self).__init__()
        self.pool2d_list = nn.ModuleList([self.make_pool(size, num_features, len(size_series)) for size in size_series])

    def make_pool(self, size, num_features, length):
        pool = nn.AdaptiveAvgPool2d(output_size=(size, size))
        # we get some trouble in bn about affine
        bn = nn.BatchNorm2d(num_features)
        relu = nn.ReLU()
        conv = nn.Conv2d(num_features, num_features/length, kernel_size=1, stride=1, bias=False)
        return nn.Sequential(pool, bn, relu, conv)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pool1 = f.upsample(input=self.pool2d_list[0](x), size=(h, w), mode='bilinear')
        pool2 = f.upsample(input=self.pool2d_list[1](x), size=(h, w), mode='bilinear')
        pool3 = f.upsample(input=self.pool2d_list[2](x), size=(h, w), mode='bilinear')
        pool6 = f.upsample(input=self.pool2d_list[3](x), size=(h, w), mode='bilinear')
        out = torch.cat((pool1, pool2, pool3, pool6, x), dim=1)
        return out


class CompressionLayer(nn.Sequential):
    def __init__(self, num_features, growth_rate):
        super(CompressionLayer, self).__init__()
        self.add_module('conv', nn.Conv2d(num_features, growth_rate, kernel_size=3, stride=1, padding=1))
        self.add_module('bn', nn.BatchNorm2d(growth_rate))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        new_features = super(CompressionLayer, self).forward(x)
        return torch.cat([x, new_features], dim=1)


class CompressionBlock(nn.Sequential):
    def __init__(self, num_features, channel=32):
        super(CompressionBlock, self).__init__()
        self.inner_features = num_features
        self.add_module('bn', nn.BatchNorm2d(self.inner_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        for i in range(2):
            layer = CompressionLayer(self.inner_features + i*channel, channel)
            self.add_module('CompressionLayer%d' % (i + 1), layer)
        self.inner_features += 2*channel
        self.add_module('CompressionLayer3.conv', nn.Conv2d(self.inner_features, self.inner_features/2, kernel_size=3, stride=1, padding=1))
        self.add_module('CompressionLayer3.bn', nn.BatchNorm2d(self.inner_features/2))
        self.add_module('CompressionLayer3.relu', nn.ReLU(inplace=True))


class Classification(nn.Sequential):
    def __init__(self, num_feature, num_classes, input_size):
        super(Classification, self).__init__()
        self.classificiation = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_feature, (num_feature/8), kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn0', nn.BatchNorm2d(num_feature/8)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(num_feature/8, num_classes, kernel_size=1, stride=1, bias=True)),
            ('dropout', nn.Dropout2d(p=0.1)),
            ('interp', nn.Upsample(size=input_size, mode='bilinear'))
        ]))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, num_layers=(6, 12, 24, 16), theta=1, num_classes=5, input_size=(512, 512), dropout_rate=0.0):
        super(DenseNet, self).__init__()

        # Convolution + pooling
        inner_feature = 2 * growth_rate
        self.Convolution = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, inner_feature, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(inner_feature)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # dense block 1
        self.denseblock1 = DenseBlock(num_layers[0], inner_feature, growth_rate, dropout_rate)
        inner_feature += num_layers[0] * growth_rate
        self.se1 = SELayer(inner_feature)
        num_features_1 = inner_feature
        self.transition1 = TransitionLayer(inner_feature, theta, stride=2, dropout_rate=dropout_rate)
        inner_feature = int(inner_feature * theta)

        # dense block 2
        self.denseblock2 = DenseBlock(num_layers[1], inner_feature, growth_rate, dropout_rate)
        inner_feature += num_layers[1] * growth_rate
        self.se2 = SELayer(inner_feature)
        num_features_2 = inner_feature
        self.transition2 = TransitionLayer(inner_feature, theta, stride=2, dropout_rate=dropout_rate)
        inner_feature = int(inner_feature * theta)

        # dense block 3
        self.denseblock3 = DenseBlock(num_layers[2], inner_feature, growth_rate, dropout_rate)
        inner_feature += num_layers[2] * growth_rate
        self.se3 = SELayer(inner_feature)
        num_features_3 = inner_feature
        self.transition3 = TransitionLayer(inner_feature, theta, stride=2, dropout_rate=dropout_rate)
        inner_feature = int(inner_feature * theta)

        # dense block 4
        self.denseblock4 = DenseBlock(num_layers[3], inner_feature, growth_rate, dropout_rate)
        inner_feature += num_layers[3] * growth_rate
        self.transition4 = TransitionLayer(inner_feature, theta, stride=1, dropout_rate=dropout_rate)
        inner_feature = int(inner_feature * theta)

        # PSP Module
        self.psp_module = PSPModule(inner_feature, [1, 2, 3, 6])
        inner_feature *= 2
        self.classifier0 = Classification(inner_feature, num_classes, input_size)

        # deconvolution 1
        self.deconv1 = nn.ConvTranspose2d(inner_feature, inner_feature/4, kernel_size=4, stride=2, padding=1, output_padding=1)
        inner_feature = inner_feature / 4 + num_features_3
        self.comp1 = CompressionBlock(inner_feature)
        inner_feature = int((inner_feature + 2*growth_rate)/2)
        self.classifier1 = Classification(inner_feature, num_classes, input_size)

        # deconvolution 2
        self.deconv2 = nn.ConvTranspose2d(inner_feature, inner_feature/4, kernel_size=4, stride=2, padding=1)
        inner_feature = inner_feature / 4 + num_features_2
        self.comp2 = CompressionBlock(inner_feature)
        inner_feature = int((inner_feature + 2*growth_rate)/2)
        self.classifier2 = Classification(inner_feature, num_classes, input_size)

        # deconvolution 3
        self.deconv3 = nn.ConvTranspose2d(inner_feature, inner_feature/4, kernel_size=4, stride=2, padding=1)
        inner_feature = inner_feature / 4 + num_features_1
        self.comp3 = CompressionBlock(inner_feature)
        inner_feature = int((inner_feature + 2*growth_rate)/2)
        self.classifier3 = Classification(inner_feature, num_classes, input_size)

    def forward(self, x):
        # down sample
        features = self.Convolution(x)
        features_1 = self.denseblock1(features)
        features = self.transition1(features_1)
        features_2 = self.denseblock2(features)
        features = self.transition2(features_2)
        features_3 = self.denseblock3(features)
        features = self.transition3(features_3)
        features = self.transition4(self.denseblock4(features))
        features = self.psp_module(features)

        # out 0
        out0 = self.classifier0(features)

        # up sample 1
        new_features = self.deconv1(features)
        new_features = torch.cat([new_features, self.se3(features_3)], dim=1)
        new_features = self.comp1(new_features)

        # out 1
        out1 = self.classifier1(new_features)

        # up sample 2
        new_features = self.deconv2(new_features)
        new_features = torch.cat([new_features, self.se2(features_2)], dim=1)
        new_features = self.comp2(new_features)

        # out 2
        out2 = self.classifier2(new_features)

        # up sample 3
        new_features = self.deconv3(new_features)
        new_features = torch.cat([new_features, self.se1(features_1)], dim=1)
        new_features = self.comp3(new_features)

        # out 3
        out3 = self.classifier3(new_features)
        return out0, out1, out2, out3
