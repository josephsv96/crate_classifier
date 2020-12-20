import numpy as np

try:
    from sliding import SlidingWindow
    from utils import load_json

except ImportError as error:
    print(f"Error: {error}; Local modules not found")
except Exception as exception:
    print(exception)

# Helper Functions


def img_init(h, w, depth):
    img_arr = np.zeros((h, w, depth))
    PKG_1_PARAMS = load_json("pkg_1_config.json")
    return img_arr, PKG_1_PARAMS

# Tests


def test_model_1():
    img_arr, params = img_init(128, 128, 1)
    sliding_obj = SlidingWindow(img_arr, w_scale=2, PARAMS=params)

    assert(sliding_obj.N_dim == 128)
    assert(sliding_obj.w_dim == 64)
    assert(sliding_obj.w_stride == 32)
    assert(sliding_obj.sliced_arr.shape == (9, 64, 64, 1))


def test_model_2():
    img_arr, params = img_init(128, 128, 1)
    sliding_obj = SlidingWindow(img_arr, w_scale=3, PARAMS=params)

    assert(sliding_obj.N_dim == 128)
    assert(sliding_obj.w_dim == 42)
    assert(sliding_obj.w_stride == 21)
    assert(sliding_obj.sliced_arr.shape == (25, 42, 42, 1))


def test_model_3():
    img_arr, params = img_init(128, 128, 3)
    sliding_obj = SlidingWindow(img_arr, w_scale=4, PARAMS=params)

    assert(sliding_obj.N_dim == 128)
    assert(sliding_obj.w_dim == 32)
    assert(sliding_obj.w_stride == 16)
    assert(sliding_obj.sliced_arr.shape == (49, 32, 32, 3))


def test_model_4():
    img_arr, params = img_init(128, 128, 3)
    sliding_obj = SlidingWindow(img_arr, w_scale=5, PARAMS=params)

    assert(sliding_obj.N_dim == 128)
    assert(sliding_obj.w_dim == 25)
    assert(sliding_obj.w_stride == 12)
    assert(sliding_obj.sliced_arr.shape == (81, 25, 25, 3))
