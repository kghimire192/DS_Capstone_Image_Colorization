# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
# from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import *
from sklearn.model_selection import train_test_split
import math
from tensorflow.python.keras.utils import losses_utils

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
tf.config.experimental.list_physical_devices('GPU')
tf.random.set_seed(42)
np.random.seed(42)

# %% ----------------------------------- Hyper Parameters and Constants ------------------------------------------------
IMG_SIZE = [224, 224]
N_CHANNELS = 3
N_CLASSES = 169
DIR_BASE = '/home/ubuntu/capstone/train_test/'
PRED_DIR = '/home/ubuntu/capstone//prediction/'
# DIR_BASE = '/Users/kanchanghimire/Development/Preprocessing/train_test/'
VALID_SIZE = 0.1765
BATCH_SIZE = 8
NUM_EPOCHS = 10

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x_train = np.load(file=DIR_BASE + "x_train.npy", allow_pickle=True)
y_train = np.load(file=DIR_BASE + "y_train.npy", allow_pickle=True)
x_test = np.load(file=DIR_BASE + "x_test.npy", allow_pickle=True)
y_test = np.load(file=DIR_BASE + "y_test.npy", allow_pickle=True)
x_pred = np.load(file=PRED_DIR + "x_pred.npy", allow_pickle=True)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=VALID_SIZE)


class CustomDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, num_classes=N_CLASSES):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.n_classes = num_classes

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        batch_y = tf.keras.utils.to_categorical(y=batch_y,
                                                num_classes=N_CLASSES)

        return batch_x, batch_y


training_generator = CustomDataGenerator(x_train, y_train, batch_size=BATCH_SIZE)
validation_generator = CustomDataGenerator(x_valid, y_valid, batch_size=BATCH_SIZE)
testing_generator = CustomDataGenerator(x_test, y_test, batch_size=BATCH_SIZE)


# %% -------------------------------------- FCN8 Class -----------------------------------------------------------------
def FCN8(image_size, ch_in, ch_out):
    inputs = Input(shape=(*image_size, ch_in), name='input')

    # Building a pre-trained VGG-16 as a feature extractor
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)

    # Feature maps of final 3 blocks:
    f3 = vgg16.get_layer('block3_pool').output  # shape: (28, 28, 256)
    f4 = vgg16.get_layer('block4_pool').output  # shape: (14, 14, 512)
    f5 = vgg16.get_layer('block5_pool').output  # shape: ( 7,  7, 512)

    f5conv1 = Conv2D(filters=4086, kernel_size=7, padding='same',
                     activation='relu')(f5)
    f5drop1 = Dropout(0.5)(f5conv1)
    f5conv2 = Conv2D(filters=4086, kernel_size=1, padding='same',
                     activation='relu')(f5drop1)
    f5drop2 = Dropout(0.5)(f5conv2)
    f5conv3 = Conv2D(filters=ch_out, kernel_size=1, padding='same',
                     activation=None)(f5drop2)

    # Upscale f5 into a 14x14 feature map
    f5conv3x2 = Conv2DTranspose(filters=ch_out, kernel_size=4, strides=2,
                                use_bias=False, padding='same',
                                activation='relu')(f5)
    f4conv1 = Conv2D(filters=ch_out, kernel_size=1, padding='same',
                     activation=None)(f4)
    m1 = add([f4conv1, f5conv3x2])

    m1x2 = Conv2DTranspose(filters=ch_out, kernel_size=4, strides=2,
                           use_bias=False, padding='same',
                           activation='relu')(m1)
    f3conv1 = Conv2D(filters=ch_out, kernel_size=1, padding='same',
                     activation=None)(f3)
    m2 = add([f3conv1, m1x2])

    # Upscale the feature map to the original shape of 224x224
    outputs = Conv2DTranspose(filters=ch_out, kernel_size=16, strides=8,
                              padding='same', activation='softmax')(m2)

    model = Model(inputs, outputs)
    return model


# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = FCN8(IMG_SIZE, N_CHANNELS, N_CLASSES)
model.summary()


def dice_loss(y_true, y_pred, eps=1e-6, spatial_axes=[1, 2], from_logits=False):
    num_classes = y_pred.shape[-1]

    # Transform logits in probabilities, and one-hot the ground-truth:

    # Compute Dice numerator and denominator:
    num_perclass = 2 * tf.math.reduce_sum(y_pred * y_true, axis=spatial_axes)
    den_perclass = tf.math.reduce_sum(y_pred + y_true, axis=spatial_axes)

    # Compute Dice and average over batch and classes:
    dice = tf.math.reduce_mean((num_perclass + eps) / (den_perclass + eps))

    return 1 - dice


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, eps=1e-6, spatial_axes=[1, 2], from_logits=False, name='loss'):
        super(DiceLoss, self).__init__(reduction=losses_utils.ReductionV2.AUTO, name=name)
        self.eps = eps
        self.spatial_axes = spatial_axes
        self.from_logits = from_logits

    def call(self, y_true, y_pred, sample_weight=None):
        return dice_loss(y_true, y_pred, eps=self.eps,
                         spatial_axes=self.spatial_axes, from_logits=self.from_logits)


accuracy = tf.metrics.Accuracy(name='acc')
meanIoU = tf.metrics.MeanIoU(num_classes=N_CLASSES, name='mIoU')
loss_c = tf.keras.losses.CategoricalCrossentropy()
loss_d = DiceLoss()
optimizer = tf.keras.optimizers.Adam()

# %% -------------------------------------- Training -------------------------------------------------------------------
model.compile(optimizer=optimizer, loss=loss_d, metrics=[accuracy, meanIoU])

history = model.fit(x=training_generator, validation_data=validation_generator,
                    epochs=NUM_EPOCHS)

# %% -------------------------------------- Testing --------------------------------------------------------------------
hist_test = model.evaluate(x=testing_generator)
print(hist_test)


# %% -------------------------------------- Prediction -----------------------------------------------------------------
def predict(images, model):
    x = images
    y_pred = np.argmax(model.predict(x), axis=-1)

    return x, y_pred


x, y_pred = predict(x_pred, model)

np.save(PRED_DIR + "x_pred_new.npy", x)
np.save(PRED_DIR + "y_pred.npy", y_pred)
