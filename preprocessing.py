# Dependencies
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Local Modules
from utils import read_image, read_cmp


def img_preprocess(IMAGE_SETS):

    # Noramlizing the images
    IMAGE_SETS = np.ndarray.astype(IMAGE_SETS, 'float32')
    IMAGE_SETS = IMAGE_SETS/255

    return IMAGE_SETS


def ann_preprocess(ANNOT_SETS, class_limit):

    y = to_categorical(ANNOT_SETS, num_classes=None,
                       dtype='float32')
    new_y = np.zeros([y.shape[0], y.shape[1], y.shape[2], class_limit])
    new_y[:, :, :, :y.shape[-1]] = y[:, :, :, :]
    return new_y


def ann_preprocess_2(arr, old_config, new_config):
    """
    To match arr which has new_config match the old_config of annotations
    """

    old_config_values = list(old_config.values())
    new_config_values = list(new_config.values())

    new_order = []
    for value in new_config_values:
        index = old_config_values.index(value)
        new_order.append(index)
    new_order = np.array(new_order)

    print("Matching the configs, new config:")
    print(new_order+1)

    # Splitting per class
    arr = ann_preprocess(arr)
    arr_matched = np.zeros(arr.shape)

    for old_index, new_index in enumerate(new_order):
        arr_matched[:, :, :, new_index] = arr[:, :, :, old_index]

    return arr_matched


def resize_arr(arr, new_width, new_height):
    """
    To resize the the input to the model
    """
    batch_size = arr.shape[0]
    channels = arr.shape[-1]
    new_arr = np.zeros([batch_size, new_width, new_height, channels])
    for i in range(batch_size):
        src = arr[i, :, :, :]
        if channels == 1:
            new_arr[i, :, :, 0] = cv2.resize(src, (new_width, new_height))
        else:
            new_arr[i, :, :, :] = cv2.resize(src, (new_width, new_height))
    return new_arr


def stack_exp(image_set):
    """Stacking different exposure levels together (back to back)
    """
    image_stack = np.zeros(
        [image_set.shape[0], image_set.shape[2], image_set.shape[3], 9])

    for i in range(image_set.shape[0]):
        index = 0
        for j in range(3):
            for k in range(3):
                # stacking same exposure, same channels
                image_stack[i, :, :, index] = image_set[i, j, :, :, k]
                index = index+1

    return image_stack


def avg_exp(image_set):
    """Averaging different exposures into 1 image
    """
    new_arr = np.zeros(
        [image_set.shape[0], image_set.shape[2], image_set.shape[3], 3])

    for i in range(image_set.shape[0]):
        for j in range(image_set.shape[1]):
            new_arr[i, :, :, :] += image_set[i, j, :, :, :]
        new_arr[i, :, :, :] /= 3

    return new_arr


def limit_output_class(y, limited_list):
    '''
    Function to limit the classes of y
    y             - original output array will all classes
    limited_list  - a list of the classes to limit
    limited_y     - output array with lmited classes
    '''
    # all_class = list(np.arange(y.shape[-1]))

    # # Selecting the list of classes to limit
    # limited_y_shape = list(y.shape[0:3])
    # limited_y_shape.append(y.shape[3] - len(limited_list))

    limited_y = np.zeros(y.shape)

    # j = 0
    for i in range(y.shape[-1]):
        if i not in limited_list:
            limited_y[:, :, :, i] = y[:, :, :, i]
            # j += 1

    return limited_y


def split_data(images, labels, RANDOM_STATE=0):
    train_images, test_images, train_labels, test_labels = train_test_split(
        images,
        labels,
        test_size=0.3,
        random_state=RANDOM_STATE)

    return [train_images, train_labels], [test_images, test_labels]


def stack_exp_v2(image_arr, num_exp=3):
    """To reshape image array [[width, height, 3] * 3] to [[width, height, 9]]
    """
    new_image_arr = np.zeros(
        [int(image_arr.shape[0]/num_exp), image_arr.shape[1], image_arr.shape[2], num_exp*3])
    j = 0
    for i in range(int(image_arr.shape[0]/num_exp)):
        k = 0
        for exp in range(num_exp):
            new_image_arr[i, :, :, k:k+3] = image_arr[j, :, :, :]
            k += 3
            j += 1
    return new_image_arr


def ann_preprocess_v2(annot_arr, num_exp=3):
    """Only the first annotation is taken from the the annot_arr
    """
    new_annot_arr = np.zeros(
        [int(annot_arr.shape[0]/3), annot_arr.shape[1], annot_arr.shape[2], annot_arr.shape[3]])
    j = 0
    for i in range(int(annot_arr.shape[0]/num_exp)):
        new_annot_arr[i, :, :, :] = annot_arr[j, :, :, :]
        j += num_exp
    return new_annot_arr


def get_dataset(dataset_path, num_exp=1):
    """Create the image and annotation dataset from given path.
    Given path should contain both image and corresponding annotation.
    Image file as .bmp.
    Annotation file as .cmp

    Args:
        dataset_path ([path]): [Path of the image and annotation files]
        num_exp (int, optional): [Number of exposures of images]. Defaults to 1.

    Returns:
        [numpy.array]: [image array, (num_img, height, width, num_exp)]
        [numpy.array]: [annotation array, (num_img, height, width, 1)]
    """
    img_files = list(dataset_path.glob("**/*.bmp"))
    cmp_files = list(dataset_path.glob("**/*.cmp"))

    # State of image directory
    img_num = int(len(img_files)/num_exp)
    err_path = dataset_path.parent.stem + "/" + dataset_path.stem
    assert(img_num == len(cmp_files)
           ), f"Number of images and annotations should be equal in {err_path}"

    # Initializing img_arr, ann_arr
    init_img = read_image(img_files[0])
    init_h = init_img.shape[0]
    init_w = init_img.shape[1]

    img_arr = np.zeros([img_num, init_h, init_w, 3 * num_exp])
    ann_arr = np.zeros([img_num, init_h, init_w, 1])

    # Loading the imgages
    j = 0
    for i in tqdm(range(img_num), desc="Creating dataset"):
        # scale this to num_exp
        img_arr[i, :, :, 0:3] = read_image(img_files[j])
        img_arr[i, :, :, 3:6] = read_image(img_files[j+1])
        img_arr[i, :, :, 6:9] = read_image(img_files[j+2])
        j += 3

        ann_arr[i, :, :, 0] = read_cmp(cmp_files[i], init_h, init_w)

    return img_arr, ann_arr
