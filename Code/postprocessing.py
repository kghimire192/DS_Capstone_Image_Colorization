import numpy as np
import cv2

PRED_DIR = '/home/ubuntu/capstone//prediction/'

x = np.load(PRED_DIR + "x_pred.npy", allow_pickle=True)
y_pred = np.load(PRED_DIR + "y_pred.npy", allow_pickle=True)

print(x.shape)
print(y_pred.shape)


# Create Color Array
# min = 0, max = 250, interval = 20
def create_color_array(min, max, interval):
    ab_color_list = []

    for i in range(min, max, interval):
        for j in range(min, max, interval):
            ab_color_list.append([i, j])
            j += interval
        i += interval

    return ab_color_list


def convert_numpy_target_array(color_array, target_numpy_array):
    target_list = target_numpy_array.tolist()
    a_list = []
    b_list = []

    for i in range(0, len(target_list)):
        new_a_list_row = []
        new_b_list_row = []
        for j in range(0, len(target_list[i])):
            a = color_array[target_list[i][j]][0]
            b = color_array[target_list[i][j]][1]
            new_a_list_row.append(a)
            new_b_list_row.append(b)
        a_list.append(new_a_list_row)
        b_list.append(new_b_list_row)

        # L, a, b require specific data types 'dtype=uint8'
        a_list_numpy_array = np.array(a_list, dtype=np.uint8)
        b_list_numpy_array = np.array(b_list, dtype=np.uint8)

    return (a_list_numpy_array, b_list_numpy_array)


def post_process_image(x, y_pred, ab_color_list, index):
    L = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    a, b = convert_numpy_target_array(ab_color_list, y_pred)
    merged_image = cv2.merge((L, a, b))
    image_bgr = cv2.cvtColor(merged_image, cv2.COLOR_LAB2BGR)
    cv2.imwrite(PRED_DIR + str(index) + '.jpg', image_bgr)


ab_color_list = create_color_array(0, 260, 20)

for i in range(len(x)):
    post_process_image(x[i], y_pred[i], ab_color_list, i)
