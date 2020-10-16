from keras.layers import Input, Dense, Add, Conv2D, Conv1D, MaxPooling2D, UpSampling2D, Activation, ZeroPadding2D, Flatten, \
    Dropout, AveragePooling2D, BatchNormalization, ReLU, Concatenate, GlobalAveragePooling2D, GlobalMaxPool2D, Lambda, \
    Reshape, merge, SpatialDropout2D, concatenate
import keras.backend as K
from keras.models import Model, Sequential
from keras.applications import DenseNet121, DenseNet169, DenseNet201, ResNet50
from keras import regularizers
from keras.utils.data_utils import get_file
from segmentation_models import Unet
import numpy as np
from keras.losses import binary_crossentropy


def build_kp_model(input_shape=(256, 256, 1), with_dropout=True):
    kwargs     = {'activation':'relu', 'padding':'same'}
    conv_drop  = 0.2
    dense_drop = 0.5
    inp        = Input(shape=input_shape)

    x = inp

    x = Conv2D(64, (9, 9), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    h = MaxPooling2D(pool_size=(1, int(x.shape[2])))(x)
    h = Flatten()(h)
    if with_dropout: h = Dropout(dense_drop)(h)
    h = Dense(16, activation='relu')(h)

    v = MaxPooling2D(pool_size=(int(x.shape[1]), 1))(x)
    v = Flatten()(v)
    if with_dropout: v = Dropout(dense_drop)(v)
    v = Dense(16, activation='relu')(v)

    x = Concatenate()([h,v])
    if with_dropout: x = Dropout(0.5)(x)
    x = Dense(16, activation='linear')(x)
    return Model(inp,x)


def build_kp_model_2(input_shape=(256, 256, 1), with_dropout=True):
    kwargs     = {'activation':'relu', 'padding':'same'}
    conv_drop  = 0.2
    dense_drop = 0.5
    inp        = Input(shape=input_shape)

    x = inp

    x = Conv2D(64, (9, 9), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    h = MaxPooling2D(pool_size=(1, int(x.shape[2])))(x)
    h = Flatten()(h)
    if with_dropout: h = Dropout(dense_drop)(h)
    h = Dense(512, activation='relu')(h)

    v = MaxPooling2D(pool_size=(int(x.shape[1]), 1))(x)
    v = Flatten()(v)
    if with_dropout: v = Dropout(dense_drop)(v)
    v = Dense(256, activation='relu')(v)

    x = Concatenate()([h,v])
    if with_dropout: x = Dropout(0.5)(x)
    x = Dense(16, activation='linear')(x)
    return Model(inp,x)


def build_kp_model_3(input_shape=(256, 256, 1), with_dropout=True):
    kwargs     = {'activation':'relu', 'padding':'same'}
    conv_drop  = 0.2
    dense_drop = 0.5
    inp        = Input(shape=input_shape)

    x = inp

    x = Conv2D(64, (9, 9), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    h = MaxPooling2D(pool_size=(1, int(x.shape[2])))(x)
    h = Flatten()(h)
    if with_dropout: h = Dropout(dense_drop)(h)
    h = Dense(16, activation='relu')(h)

    v = MaxPooling2D(pool_size=(int(x.shape[1]), 1))(x)
    v = Flatten()(v)
    if with_dropout: v = Dropout(dense_drop)(v)
    v = Dense(16, activation='relu')(v)

    x = Concatenate()([h,v])
    if with_dropout: x = Dropout(0.5)(x)
    x = Dense(16, activation='sigmoid')(x)
    return Model(inp,x)


def build_kp_model_4(input_shape=(256, 256, 1)):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(16))

    return model


def rn50_1(input_shape=(256, 256, 3), freeze_backbone=False):
    # build model
    base_model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)

    if freeze_backbone:
        for layer in base_model.layers:
            # if layer.name.find('bn') != -1:
            #     layer.momentum = 1
            layer.trainable = False

    x = AveragePooling2D((7, 7))(base_model.output)
    x = Flatten()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(16)(x)

    model = Model(inputs=[base_model.input], outputs=[x])
    return model


def bce_dice_loss_2ch(y_true, y_pred):
    dice0 = dice_coef_loss(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    dice1 = dice_coef_loss(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    return binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0]) + dice0 + dice1


def bce_dice_loss_double(y_true, y_pred):
    dice0 = dice_coef_loss(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    dice1 = dice_coef_loss(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    return 0.5 * binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0]) + 0.5 * (dice0 + dice1)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def dice_coef_loss_double(y_true, y_pred):
    dice0 = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    dice1 = dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])

    return 1-(0.7 * dice0 + 0.3 * dice1)


def euclidean_loss(x, y):
    return K.sqrt(K.sum(K.square(x - y)))


def down(filters, input_, dropout=None):
    down_ = Conv2D(filters, (3, 3), padding='same')(input_)
    down_ = BatchNormalization(epsilon=1e-4)(down_)
    down_ = Activation('relu')(down_)
    down_ = Conv2D(filters, (3, 3), padding='same')(down_)
    down_ = BatchNormalization(epsilon=1e-4)(down_)
    down_res = Activation('relu')(down_)
    down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_)

    if dropout is not None:
        down_pool = Dropout(dropout)(down_pool)

    return down_pool, down_res


def up(filters, input_, down_, dropout=None):
    up_ = UpSampling2D((2, 2))(input_)
    up_ = concatenate([down_, up_], axis=3)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)

    if dropout is not None:
        up_ = Dropout(dropout)(up_)

    return up_


def get_unet_256(input_shape=(256, 256, 1), num_classes=7):
    inputs = Input(shape=input_shape)

    down0b, down0b_res = down(8, inputs)
    down0a, down0a_res = down(16, down0b)
    down0, down0_res = down(32, down0a)
    down1, down1_res = down(64, down0)
    down2, down2_res = down(128, down1)
    down3, down3_res = down(256, down2)

    center = Conv2D(256, (3, 3), padding='same')(down3)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)
    center = Conv2D(256, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up3 = up(256, center, down3_res)
    up2 = up(128, up3, down2_res)
    up1 = up(64, up2, down1_res)
    up0 = up(32, up1, down0_res)
    up0a = up(16, up0, down0a_res)
    up0b = up(8, up0a, down0b_res)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_layer')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    return model


def get_unet_96(input_shape=(96, 96, 1), num_classes=7, dropout=None):
    inputs = Input(shape=input_shape)

    down0b, down0b_res = down(8, inputs, dropout)
    down0a, down0a_res = down(16, down0b, dropout)
    down0, down0_res = down(32, down0a, dropout)

    center = Conv2D(64, (3, 3), padding='same')(down0)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)
    center = Conv2D(64, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up0 = up(32, center, down0_res, dropout)
    up0a = up(16, up0, down0a_res, dropout)
    up0b = up(8, up0a, down0b_res, dropout)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_layer')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    return model


def get_unet_128(input_shape=(128, 128, 1), num_classes=2):
    inputs = Input(shape=input_shape)

    down0b, down0b_res = down(8, inputs)
    down0a, down0a_res = down(16, down0b)
    down0, down0_res = down(32, down0a)
    down1, down1_res = down(64, down0)

    center = Conv2D(128, (3, 3), padding='same')(down1)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)
    center = Conv2D(128, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up1 = up(64, center, down1_res)
    up0 = up(32, up1, down0_res)
    up0a = up(16, up0, down0a_res)
    up0b = up(8, up0a, down0b_res)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_layer')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    return model


def get_unet_custom(input_shape=(96, 96, 1), num_classes=7):
    inputs = Input(shape=input_shape)

    down0b, down0b_res = down(16, inputs)
    down0a, down0a_res = down(16, down0b)
    down0, down0_res = down(32, down0a)

    center = Conv2D(64, (3, 3), padding='same')(down0)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)
    center = Conv2D(64, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up0 = up(32, center, down0_res)
    up0a = up(16, up0, down0a_res)
    up0b = up(16, up0a, down0b_res)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_layer')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    return model


def get_unet_double(input_shape=(96, 96, 1), num_classes=7):
    inputs = Input(shape=input_shape)

    down0b, down0b_res = down(8, inputs)
    down0a, down0a_res = down(16, down0b)
    down0, down0_res = down(32, down0a)

    center = Conv2D(64, (3, 3), padding='same')(down0)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)
    center = Conv2D(64, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    x = Flatten()(center)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    points = Dense(2)(x)

    up0 = up(32, center, down0_res)
    up0a = up(16, up0, down0a_res)
    up0b = up(8, up0a, down0b_res)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_layer')(up0b)

    model = Model(inputs=inputs, outputs=[classify, points])

    return model


def get_unet_96_dense(input_shape=(96, 96, 1), output_number=14):
    inputs = Input(shape=input_shape)

    down0b, down0b_res = down(8, inputs)
    down0a, down0a_res = down(16, down0b)
    down0, down0_res = down(32, down0a)

    center = Conv2D(64, (3, 3), padding='same')(down0)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)
    center = Conv2D(64, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up0 = up(32, center, down0_res)
    up0a = up(16, up0, down0a_res)
    up0b = up(8, up0a, down0b_res)

    mask = Conv2D(1, (1, 1), activation='sigmoid', name='dice_output')(up0b)

    x = Flatten()(mask)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    points = Dense(output_number)(x)

    model = Model(inputs=inputs, outputs=[mask, points])
    return model


def build_model_resnet18(freeze_encoder=True, weights='imagenet'):
    model = Unet(backbone_name='resnet18', encoder_weights=weights, freeze_encoder=freeze_encoder, classes=7)
    return model


def build_color_model_1(input_shape=(64, 64, 3)):
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation="softmax"))

    return model


def build_color_model_2(input_shape=(64, 64, 3)):
    model = Sequential()

    model.add(Dense(16, activation="relu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation="softmax"))

    return model


def build_color_sides_model_1(input_shape=(128, 128, 3)):
    input = Input(input_shape)

    x = BatchNormalization(axis=-1)(input)
    x = Conv2D(16, (5, 5))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(16, (5, 5))(x)
    x = ReLU()(x)

    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(32, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(32, (3, 3))(x)
    x = ReLU()(x)

    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3))(x)
    x = ReLU()(x)

    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(32)(x)
    x = Dropout(0.5)(x)

    outputs = []
    for i in range(3):
        for j in range(3):
            outputs.append(Dense(6, activation="softmax", name='o%d%d' % (i, j))(x))

    model = Model(input, outputs)

    return model