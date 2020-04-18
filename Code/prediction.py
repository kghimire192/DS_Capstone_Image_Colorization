print("BEGIN PREPROCESSING")
# Imports
import os
import cv2
# from google.colab.patches import cv2_imshow
import numpy as np
from sklearn.model_selection import train_test_split

#DIR_BASE = '/Users/kanchanghimire/Development/Preprocessing/train_test/'

DATA_DIR = "/Users/kanchanghimire/Development/Preprocessing/prediction/"
IMG_WIDTH = 224
IMG_HEIGHT = 224
#TEST_SIZE = 0.15

images_list = list()

for file in os.listdir(DATA_DIR):
    if file.endswith('.jpg'):
        images_list.append(file)

print(len(images_list))


# Find color array index
# Interval == 20
def calculate_color_index_interval_20(a, b):
    if (a == 250):
        a = 240
    if (b == 250):
        b = 240
    index = round((a / 20), 0) * 13 + round((b / 20), 0)
    return int(index)


# Round Function
def round_numpy_array(numpy_array):
    new_rounded_list = []

    for row in numpy_array:
        new_row = []
        for element in row:
            new_row.append(round(element, -1))
        new_rounded_list.append(new_row)

    return new_rounded_list


# Round and calculate (For Interval of 20)
def round_and_calculate_interval_20(a_list, b_list):
    rounded_a_list = round_numpy_array(a_list.tolist())
    rounded_b_list = round_numpy_array(b_list.tolist())
    index_array = []

    for i in range(0, len(rounded_a_list)):
        new_row = []
        for j in range(0, len(rounded_a_list[i])):
            index = calculate_color_index_interval_20(rounded_a_list[i][j], rounded_b_list[i][j])
            new_row.append(index)
        index_array.append(new_row)

    return index_array


# For Interval 20
def process_image_interval_20(image_path, width, height):
    img = cv2.imread(image_path)  # read image
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB
    resized_img = cv2.resize(img_lab, (width, height))  # resize image

    # split into L, a and b
    L_array = resized_img[:, :, 0]
    a_array = resized_img[:, :, 1]
    b_array = resized_img[:, :, 2]

    target_array = round_and_calculate_interval_20(a_array, b_array)

    # NumPy data type needs to be 'dtype=np.uint16' or higher
    numpy_target_array = np.array(target_array, dtype=np.uint16)

    return (L_array, numpy_target_array)


def prepare_data(images_list, width, height):
    x_list = list()
    y_list = list()

    for image in images_list:
        image_path = DATA_DIR + image
        x, y = process_image_interval_20(image_path, width, height)
        x_BGR = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)  # cloned 1 channel to 3 channel for model input
        x_list.append(x_BGR)
        y_list.append(y)

    np.save(DATA_DIR + "x_pred.npy", x_list)
    np.save(DATA_DIR + "y_true.npy", y_list)


prepare_data(images_list, IMG_WIDTH, IMG_HEIGHT)

print("END PREPROCESSING")
