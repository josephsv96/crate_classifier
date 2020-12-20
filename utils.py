import numpy as np
import cv2
from pathlib import Path
from json import load as json_load
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from datetime import datetime


def get_timestamp():
    now_d = datetime.now().date()
    now_t = datetime.now().time()
    timestamp = f"{now_d}-{now_t.hour}-{now_t.minute}-{now_t.second}"
    # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    return timestamp


def read_image(img_file, height=None):
    """Return an image as numpy array for a image file using cv2

    Args:
        img_file (path): Image file path
        height (int, optional): Ouput height if desired,
                                width is scaled without loss of resolution.
                                Defaults to None.

    Returns:
        [numpy.array]: Single image as numpy array
    """

    src = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)

    if height is None:
        height = src.shape[0]

    # rescaling without loosing aspect ratio
    width = int(height * (src.shape[1] / src.shape[0]))
    src = cv2.resize(src, (width, height))

    return src


def read_cmp(cmp_file, annot_shape):

    # open binary file
    file = open(cmp_file, "rb")

    # get file size
    file.seek(0, 2)    # set pointer to end of file
    nb_bytes = file.tell()
    file.seek(0, 0)    # set pointer to begin of file
    buf = file.read(nb_bytes)
    file.close()

    # convert from byte stream to numpy array
    ann_arr = np.asarray(list(buf), dtype=np.byte)

    # reshaping to required shape
    ann_arr = ann_arr.reshape(annot_shape)

    return ann_arr


def write_cmp(output_path, annotation):
    """Write numpy array of shape (height, width, 1) to .cmp file

    Args:
        annotation ([numpy.array]): Annotation file
    """


def write_image(img, file_name):

    try:
        cv2.imwrite(file_name, img)
    except Exception:
        pass  # or you could use 'continue'

    return True


def create_output_folder(working_dir, folder_name="output_"):

    # output_dir = working_dir.parent / (working_dir.stem + "_processed")
    output_dir = working_dir / folder_name
    if output_dir.is_dir():
        try:
            # Will not execute if folder is non-empty
            output_dir.rmdir()
            output_dir.mkdir()
        except OSError as err:
            print("Error: %s : %s" % (output_dir, err.strerror))
            print("Overwriting...")
    else:
        output_dir.mkdir()

    return output_dir


def get_stem(dict_list):
    return dict_list.get('stem_2')


def sort_path(path_list):
    """Sorts a list of paths alphanumerically"""
    print("Sorting Started")
    new_path_list = []
    for path in path_list:
        new_path_list.append({'parent': str(path.parent),
                              'stem_1': path.stem.split('_')[0],
                              'stem_2': int(path.stem.split('_')[1]),
                              'suffix': path.suffix})
    new_path_list.sort(key=get_stem)

    path_list = []
    for path in new_path_list:
        new_path = []
        new_path.append(
            (path['parent'] + '\\' + path['stem_1'])
            + '_' + str(path['stem_2']) + path['suffix'])
        path_list.append(Path(''.join(new_path)))

    print("Sorting End")
    return path_list


def load_txt(working_dir):
    """To read txt bounding box labels"""
    input_path = working_dir
    files = sort_path(list(input_path.glob('**/*.txt')))
    files_num = len(list(files))
    print('Found', files_num, '.txt files')
    new_arr = []
    for txt_path in files:
        with open(txt_path, 'r') as txt_file:
            file_str = txt_file.read()
            file_arr = file_str.split('\n')
            file_arr2 = [parameters.split(' ') for parameters in file_arr]
            file_arr2.pop()
            np_array = np.array(file_arr2).astype('float32')
            new_arr.append(np_array)
    new_arr = np.array(new_arr)
    # print(new_arr.shape)

    return new_arr


def save_npy(working_dir, label_arr):
    # Destination directory of output
    output_path = working_dir / 'output'
    if output_path.is_dir():
        print("output folder exists; overwriting existing files in folder")
    else:
        print("output folder created")
        output_path.mkdir()

    # Saving as .npy files
    np.save(output_path / 'labels', label_arr)
    return None


def save_npy_v2(npy_arr, output_path):

    # Saving as .npy files
    np.save(output_path, npy_arr)
    return None


def load_npy(npy_file):
    npy_arr = np.load(npy_file)

    return npy_arr


def resize_arr(img_inst, new_shape):
    """Outputs resized image instance

    Args:
        img_inst (np.array): Input image instance
        new_shape (tuple): Output size (width, height)

    Returns:
        np.array: Resized image
    """
    resized_img = cv2.resize(img_inst, new_shape)
    return resized_img


def img_arr_to_gray(img_arr):
    gray_arr = []
    for i in range(img_arr.shape[0]):
        img_set = img_arr[i]
        gray_img = img_set.sum(axis=-1) / 255 * img_arr.shape[-1]
        gray_arr.append(gray_img)
    gray_arr = np.expand_dims(np.asarray(gray_arr), axis=-1)
    return gray_arr


def get_custom_cmap():
    # Colormap object for custom colours
    class_colors = [(0, 0, 0),
                    (0, 0.5, 0), (0, 1, 1), (0.5, 0.5, 0.5), (0.5, 1, 0.5),
                    (0.5, 0.25, 0.25), (0.5, 0, 1), (0, 0.25, 0.5), (0, 0, 1),
                    (1, 1, 1), (1, 0, 0.5)]

    CMAP_11 = LinearSegmentedColormap.from_list("cmap_11", class_colors, N=11)

    return CMAP_11


def load_json(json_file):
    with open(json_file, 'r') as fp:
        file_dict = json_load(fp)

    return file_dict


def bgr_to_rgb(img_arr, num_exp):
    buffer_arr = np.zeros((img_arr.shape))
    j = 0
    for i in range(num_exp):
        buffer_arr[:, :, :, j] = img_arr[:, :, :, j+2]
        buffer_arr[:, :, :, j+1] = img_arr[:, :, :, j+1]
        buffer_arr[:, :, :, j+2] = img_arr[:, :, :, j]
        j += 3

    return buffer_arr


def bgr_to_rgb_img(image_arr):
    image_arr_rgb = np.zeros(image_arr.shape)
    image_arr_rgb[:, :, 0] = image_arr[:, :, 2]
    image_arr_rgb[:, :, 1] = image_arr[:, :, 1]
    image_arr_rgb[:, :, 2] = image_arr[:, :, 0]

    return image_arr_rgb


def show_dataset(img_arr, ann_arr, show_num, num_exp, num_class, BGR=True):
    if BGR is True:
        img_arr = bgr_to_rgb(img_arr, num_exp)

    j = 1  # plot num counter
    plt.figure(figsize=(12, 3*show_num))
    for i in range(show_num):
        k = 0
        for exp in range(num_exp):
            plt.subplot(show_num, num_exp+1, j)
            plt.imshow(img_arr[i, :, :, k:k+num_exp]/255)
            plt.xticks([])
            plt.yticks([])
            plt.title(f"img_{i:06d}_{chr(ord('`')+exp+1)}", y=-0.15)
            k += 3
            j += 1

        plt.subplot(show_num, num_exp+1, j)
        plt.imshow(ann_arr[i, :, :, 0], vmin=0, vmax=num_class,
                   cmap=get_custom_cmap())
        plt.xticks([])
        plt.yticks([])
        plt.title(f"annot_{i:06d}", y=-0.15)
        j += 1
    # plt.tight_layout()
    plt.show()


def rm_duplicate(seq):
    """
    Remove duplicates from a list preserving order
    https://www.peterbe.com/plog/uniqifiers-benchmark
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
