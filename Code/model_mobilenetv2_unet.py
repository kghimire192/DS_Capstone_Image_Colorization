# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import *
from sklearn.model_selection import train_test_split
import math
from tensorflow.python.keras.utils import losses_utils
import segmentation_models as sm

sm.set_framework('tf.keras')
print(sm.__version__)

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

VALID_SIZE = 0.1765
BATCH_SIZE = 8
NUM_EPOCHS = 10

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x_train = np.load(file=DIR_BASE + "x_train.npy", allow_pickle=True)
y_train = np.load(file=DIR_BASE + "y_train.npy", allow_pickle=True)
x_test = np.load(file=DIR_BASE + "x_test.npy", allow_pickle=True)
y_test = np.load(file=DIR_BASE + "y_test.npy", allow_pickle=True)
x_true = np.load(file=PRED_DIR + "x_true.npy", allow_pickle=True)

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

# %% -------------------------------------- UNet Model -----------------------------------------------------------------
unet_model = sm.Unet(backbone_name='mobilenetv2', encoder_weights='imagenet',
                     classes=169, input_shape=(224, 224, 3), activation='softmax')
unet_model.summary()


# %% -------------------------------------- Training Prep ----------------------------------------------------------

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
unet_model.compile(optimizer=optimizer, loss=loss_d, metrics=[meanIoU])
history = unet_model.fit(x=training_generator,
                         validation_data=validation_generator,
                         epochs=NUM_EPOCHS)

# %% -------------------------------------- Testing --------------------------------------------------------------------
history_test = unet_model.evaluate(x=testing_generator)
print(history_test)

# %% -------------------------------------- Prediction -----------------------------------------------------------------
x_true = np.load(file=PRED_DIR + "x_true.npy", allow_pickle=True)


def predict(images, model):
    x_true = images
    y_pred = np.argmax(model.predict(x_true), axis=-1)

    return x_true, y_pred


x_pred, y_pred = predict(x_true, unet_model)

np.save(PRED_DIR + "x_pred.npy", x_true)
np.save(PRED_DIR + "y_pred.npy", y_pred)
