"""Tests for pkg_1a.
Assumes img_file and ann_file follow naming conventions; see README.md
"""
import numpy as np
from pathlib import Path

try:
    from tests.testing_utils import load_params_1

except ImportError as error:
    print(f"Error: {error}; Local modules not found")
except Exception as exception:
    print(exception)


# Tests


# def test_read_cmp():
#     PARAMS = load_params_1()
#     img_h, img_w = PARAMS["img_src_shape"]
#     src_dir = PARAMS["src_dir"]

#     all_cmp_files = list(Path(src_dir).glob("**/*.cmp"))

#     cmp_file = all_cmp_files[0]
#     # open binary file
#     file = open(cmp_file, "rb")

#     # get file size
#     file.seek(0, 2)    # set pointer to end of file
#     nb_bytes = file.tell()
#     file.seek(0, 0)    # set pointer to beginning of file
#     buf = file.read(nb_bytes)
#     file.close()

#     # convert from byte stream to numpy array
#     ann_arr = np.asarray(list(buf), dtype=np.byte)

#     # reshaping to required shape
#     ann_arr = ann_arr.reshape((img_h, img_w))

#     # trying to write to .cmp file
#     ann_arr_flat = np.byte(ann_arr.flatten())
#     ann_bytearray = ann_arr_flat.tobytes()

#     assert(buf == ann_bytearray)

#     # img_files = data_checker_obj.img_paths
#     # ann_files = data_checker_obj.ann_paths
#     # assert(int(len(img_files)/num_exp) == len(ann_files))
#     # assert(img_files[int(0*num_exp)].stem.split('_')[1] ==
#     #        ann_files[0].stem.split('_')[1])


# test_read_cmp()
