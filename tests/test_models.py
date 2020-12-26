"""Tests for models.
Assumes img_file and ann_file follow naming conventions; see README.md
"""
try:
    from src.models import model_14k
    from src.models import model_densenet21
    from src.models import model_densenet21_35k
    from src.models import model_densenet21_38k
    from src.models import model_densenet21_48k
    from src.models import model_densenet21_53k
    from src.models import model_resnet18
    from src.models import model_resnet18_34k
    from src.models import model_resnet18_46k
    from src.models import model_resnet18_58k
    from src.models import model_resnet18_70k
    from src.models import model_vgg16
    from src.models import model_vgg16_24k
    from src.models import model_vgg16_34k
    from src.models import model_vgg16_47k
    from src.models import model_vgg16_51k
    from src.models import model_vgg16_64k

except ImportError as error:
    print(f"Error: {error}; Local modules not found")
except Exception as exception:
    print(exception)

# Test Constants

h, w, exp, d, class_num, lr, ep = (128, 128, 3, 20, 3, 0.01, 100)

# Base Models


def test_model_1():
    model = model_14k.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 14000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")

# DenseNets


def test_model_2():
    model = model_densenet21.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 133000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_3():
    model = model_densenet21_35k.CrateNet.build(
        h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 35000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_4():
    model = model_densenet21_38k.CrateNet.build(
        h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 38000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_5():
    model = model_densenet21_48k.CrateNet.build(
        h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 48000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_6():
    model = model_densenet21_53k.CrateNet.build(
        h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 53000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")

# ResNets


def test_model_7():
    model = model_resnet18.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 182000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_8():
    model = model_resnet18_34k.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 34000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_9():
    model = model_resnet18_46k.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 46000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_10():
    model = model_resnet18_58k.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 58000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_11():
    model = model_resnet18_70k.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 70000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")

# VGG


def test_model_12():
    model = model_vgg16.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 160000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_13():
    model = model_vgg16_24k.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 24000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_14():
    model = model_vgg16_34k.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 34000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_15():
    model = model_vgg16_47k.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 47000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_16():
    model = model_vgg16_51k.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 51000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")


def test_model_17():
    model = model_vgg16_64k.CrateNet.build(h, w, exp, d, class_num, lr, ep)
    model.summary()
    assert(model.count_params() > 64000)
    assert(model.layers[0].name == "input_1")
    assert(model.layers[-1].name == "reshape_1")
