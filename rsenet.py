from typing import Tuple

from keras import layers as L
from keras.models import Model


def BasicBlock(x, out_channels, strides=1, base_width=64, groups=1, activation=L.ReLU) -> L.Layer:
    h = x

    x = L.Conv2D(out_channels, 3, strides=strides, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = activation()(x)
    x = L.Conv2D(out_channels, 3, strides=strides, padding='same')(x)
    x = L.BatchNormalization()(x)

    if h.shape[-1] != x.shape[-1]:
        h = L.Conv2D(out_channels, 1, strides=strides, padding='same')(h)

    x = L.add([x, h])
    x = activation()(x)

    return x


def Bottleneck(x, out_channels, strides=1, base_width=64, groups=1, activation=L.ReLU) -> L.Layer:
    def _grouped_CNN2D(x, out_channels, groups, strides=1):
        if groups == 1:
            # group == 1: normal CNN
            return L.Conv2D(out_channels, 3, strides=strides, padding='same')(x)
        else:
            assert x.shape[-1] % groups == 0
            assert out_channels % groups == 0
            grouped_in_channels = x.shape[-1] // groups
            grouped_out_channels = out_channels // groups
            group_list = []

            for g in range(groups):
                h = L.Lambda(lambda z: z[:, :, :, g * grouped_in_channels:(g + 1) * grouped_in_channels])(x)
                h = L.Conv2D(grouped_out_channels, 3, strides=strides, padding='same', use_bias=False)(h)
                group_list.append(h)

            return L.concatenate(group_list, axis=-1)

    width = int(out_channels * (base_width / 64.)) * groups
    h = x

    x = L.Conv2D(width, 1, strides=strides, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = _grouped_CNN2D(x, width, groups, strides=strides)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(out_channels * 4, strides=strides, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = activation()(x)

    if h.shape[-1] != x.shape[-1]:
        h = L.Conv2D(out_channels * 4, 1, strides=strides, padding='same')(h)

    x = L.add([x, h])
    x = activation()(x)

    return x


def ResNet(resnet_name: str, input_shape: Tuple[int, int, int], num_classes=1000, activation=L.ReLU) -> Model:
    """

    :param resnet_name:
    :param input_shape:
    :param num_classes:
    :param activation:
    :return:
    """
    RESNET_NAMES = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                    'resnext50_32x4d', 'resnext101_32x8d',
                    'wide_resnet50_2', 'wide_resnet101_2']
    assert resnet_name in RESNET_NAMES, f'resnet_name must be one of {RESNET_NAMES}'
    RESNET_BLOCKS = {
        'resnet18': BasicBlock,
        'resnet34': BasicBlock,
        'resnet50': Bottleneck,
        'resnet101': Bottleneck,
        'resnet152': Bottleneck,
        'resnext50_32x4d': Bottleneck,
        'resnext101_32x8d': Bottleneck,
        'wide_resnet50_2': Bottleneck,
        'wide_resnet101_2': Bottleneck
    }
    RESNET_LAYERS = {
        'resnet18': (2, 2, 2, 2),
        'resnet34': (3, 4, 6, 3),
        'resnet50': (3, 4, 6, 3),
        'resnet101': (3, 4, 23, 3),
        'resnet152': (3, 8, 36, 3),
        'resnext50_32x4d': (3, 4, 6, 3),
        'resnext101_32x8d': (3, 4, 23, 3),
        'wide_resnet50_2': (3, 4, 6, 3),
        'wide_resnet101_2': (3, 4, 23, 3)
    }
    RESNET_WIDTHS = {
        'resnet18': 64,
        'resnet34': 64,
        'resnet50': 64,
        'resnet101': 64,
        'resnet152': 64,
        'resnext50_32x4d': 128,
        'resnext101_32x8d': 256,
        'wide_resnet50_2': 128,
        'wide_resnet101_2': 128
    }
    RESNET_GROUPS = {
        'resnet18': 1,
        'resnet34': 1,
        'resnet50': 1,
        'resnet101': 1,
        'resnet152': 1,
        'resnext50_32x4d': 32,
        'resnext101_32x8d': 32,
        'wide_resnet50_2': 1,
        'wide_resnet101_2': 1
    }

    block = RESNET_BLOCKS[resnet_name]
    layers = RESNET_LAYERS[resnet_name]
    width = RESNET_WIDTHS[resnet_name]
    groups = RESNET_GROUPS[resnet_name]

    def _make_layer(x, out_channels, blocks, stride=1):
        for _ in range(blocks):
            x = block(x, out_channels, stride=stride, base_width=width, groups=groups, activation=activation)
        return x

    inputs = L.Input(shape=input_shape)
    x = L.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)

    x = L.BatchNormalization()(x)
    x = activation()(x)
    x = L.MaxPool2D()(x)

    x = _make_layer(x, 64, layers[0], stride=1)
    x = _make_layer(x, 128, layers[1], stride=2)
    x = _make_layer(x, 256, layers[2], stride=2)
    x = _make_layer(x, 512, layers[3], stride=2)

    x = L.GlobalAveragePooling2D()(x)
    x = L.Flatten()(x)
    x = L.Dense(num_classes, activation='softmax')(x)

    return x
