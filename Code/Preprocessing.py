import os
from PIL import Image

DIR_BASE = "/Users/kanchanghimire/Development/Capstone/DS_Capstone_Image_Colorization/Data"


def get_file_names(dir_name):
    images_list = list()

    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg'):
                # file_name, ext = file.split('.')
                # images_list.append(file_name)
                images_list.append(file)
    return images_list


def show_image(image_path):
    im = Image.open(image_path)
    im.show()


images_names_list = get_file_names(DIR_BASE)
# print(images_names_list)
print(len(images_names_list))

image_path = DIR_BASE + '/Places365_test_00001497.jpg'
# show_image(image_path)


from basic_image_eda import BasicImageEDA

if __name__ == "__main__":  # for multiprocessing
    data_dir = "/Users/kanchanghimire/Development/Capstone/DS_Capstone_Image_Colorization/Sample"

    # below are default values.
    extensions = ['png', 'jpg', 'jpeg']
    threads = 0
    dimension_plot = True
    channel_hist = False
    nonzero = False

    BasicImageEDA.explore(data_dir, extensions, threads, dimension_plot, channel_hist, nonzero)



