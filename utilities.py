import numpy as np
import cv2
from pathlib import Path


def read_image(img_file, height=512):

    src = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
    # rescaling without loosing aspect ratio

    width = int(height * (src.shape[1] / src.shape[0]))
    src = cv2.resize(src, (width, height))

    return src


def write_image(img, file_name):

    try:
        cv2.imwrite(file_name, img)
    except Exception:
        pass  # or you could use 'continue'

    return True


def create_output_folder(working_dir):

    output_dir = working_dir.parent / (working_dir.stem + "_processed")
    if output_dir.is_dir():
        print("output folder exists")
        try:
            output_dir.rmdir()
        except OSError as err:
            print("Error: %s : %s" % (output_dir, err.strerror))
            print("Overwriting...")
    else:
        print("output folder created")
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


def resize_arr(img_arr, new_width, new_height):
    resized_arr = cv2.resize(img_arr, (new_width, new_height))
    return resized_arr
