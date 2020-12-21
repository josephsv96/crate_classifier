"""Tests for pkg_1a.
Assumes img_file and ann_file follow naming conventions; see README.md
"""
from pathlib import Path

try:
    from modules.pkg_1a import DataChecker
    from utils import load_json

except ImportError as error:
    print(f"Error: {error}; Local modules not found")
except Exception as exception:
    print(exception)

# Helper Functions


def load_params():
    """Returns source path of images and number of exposures
    """
    PKG_1_PARAMS = load_json("pkg_1_config.json")

    return PKG_1_PARAMS["src_dir"], PKG_1_PARAMS["num_exp"]


def chk_dir(src_dir):
    """To check if the directory exists. Returns boolean
    """
    src_is_dir = Path(src_dir).is_dir()

    if src_is_dir is False:
        print(f"Test Directory '{src_dir}' does not exist.")
        print("Add a directory containing the source images to run this test")

    return src_is_dir

# Tests


def test_data_checker_1():
    src_dir, num_exp = load_params()
    assert(chk_dir(src_dir) is True)

    data_checker_obj = DataChecker(src_dir, num_exp)
    img_files = data_checker_obj.img_paths
    ann_files = data_checker_obj.ann_paths
    assert(int(len(img_files)/num_exp) == len(ann_files))
    assert(img_files[0].stem == ann_files[0].stem)
    assert(img_files[int(0*num_exp)].stem.split('_')[1] ==
           ann_files[0].stem.split('_')[1])


def test_data_checker_2():
    src_dir, num_exp = load_params()
    assert(chk_dir(src_dir) is True)

    data_checker_obj = DataChecker(src_dir, num_exp)
    img_files = data_checker_obj.img_paths
    ann_files = data_checker_obj.ann_paths
    assert(int(len(img_files)/num_exp) == len(ann_files))
    assert(img_files[0].stem == ann_files[0].stem)
    assert(img_files[int(10*num_exp)].stem.split('_')[1] ==
           ann_files[10].stem.split('_')[1])
