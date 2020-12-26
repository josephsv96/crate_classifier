import numpy as np
from data_loader import load_json, load_npy
from preprocessing import img_preprocess, ann_preprocess
from preprocessing import resize_arr, split_data, stack_exp
from model import make_model, make_model_2


def train_model_1(img_arr, annot_arr, net_h, net_w, class_limit, epochs=100):
    """
    To train the standard model.
    img_arr shape => (img_num, height, width, 9)
    """
    # Preprocessing - Images
    X = resize_arr(img_preprocess(img_arr), net_h, net_w)
    print("X shape:", X.shape)

    # Preprocessing - Annotations
    y = resize_arr(ann_preprocess(annot_arr, class_limit),
                   net_h, net_w)
    print("y shape:", y.shape)

    # Splitting data
    train_data, test_data = split_data(X, y)

    model_input, model = make_model(img_shape=[net_w, net_h],
                                    num_classes=class_limit,
                                    num_exposures=3)

    model_hist = model.fit(x=train_data[0],
                           y=train_data[1],
                           validation_data=test_data,
                           epochs=epochs)

    return model, model_input, model_hist


def add_is_crate(class_label_arr):
    is_crate_arr = class_label_arr[:, :, :, 0] - (1.0) * (-1)
    is_crate_arr_2 = np.zeros(
        [is_crate_arr.shape[0], is_crate_arr.shape[1], is_crate_arr.shape[2], 1])
    is_crate_arr_2[:, :, :, 0] = is_crate_arr
    return is_crate_arr_2


def train_model_1_v2(img_arr, annot_arr, net_h, net_w, class_limit, init_lr=1e-3, epochs=100):
    """
    To train the standard model.
    img_arr shape => (img_num, height, width, 9)
    """
    # Preprocessing - Images
    X = resize_arr(img_preprocess(img_arr), net_h, net_w)
    print("X shape:", X.shape)

    # Preprocessing - Annotations
    y = resize_arr(ann_preprocess(annot_arr, class_limit),
                   net_h, net_w)
    print("y shape:", y.shape)

    # Splitting data
    train_data, test_data = split_data(X, y)

    # Creating 'isCrate output'
    train_data.append(add_is_crate(train_data[1]))
    test_data.append(add_is_crate(test_data[1]))

    # Making the model
    model_input, model = make_model_2(img_shape=[net_w, net_h],
                                      num_classes=class_limit,
                                      num_exposures=3,
                                      init_lr=init_lr,
                                      epochs=epochs)

    model_hist = model.fit(x=train_data[0],
                           y={"isCrate": train_data[2],
                               "class": train_data[1]},
                           validation_data=(test_data[0],
                                            {"isCrate": test_data[2],
                                             "class": test_data[1]}),
                           epochs=epochs)

    return model, model_input, model_hist


def train_model_2(img_arr, annot_arr, net_h, net_w, class_limit, epochs=100):
    """
    To train the standard model.
    img_arr shape => (img_num, height, width, 9)
    train_model_1 adding "is_crate" class
    """
    # Preprocessing - Images
    X = resize_arr(img_preprocess(img_arr), net_h, net_w)
    print("X shape:", X.shape)

    # Preprocessing - Annotations
    y = resize_arr(ann_preprocess(annot_arr, class_limit),
                   net_h, net_w)

    # Adding "is_crate" class
    y[:, :, :, 0] = (y[:, :, :, 0] * -1) + 1
    print("y shape:", y.shape)

    # Splitting data
    train_data, test_data = split_data(X, y)

    model_input, model = make_model(img_shape=[net_w, net_h],
                                    num_classes=class_limit,
                                    num_exposures=3)

    model_hist = model.fit(x=train_data[0],
                           y=train_data[1],
                           validation_data=test_data,
                           epochs=epochs)

    return model, model_input, model_hist


def main():
    data_dir = "dataset/data_1"
    images_file = data_dir + "/dataset_images.npy"
    labels_file = data_dir + "/dataset_labels.npy"
    config = data_dir + "/label_config.json"

    images = load_npy(images_file)
    annotations = load_npy(labels_file)
    config = load_json(config)

    print("Image batch shape:\t", images.shape)
    print("Annotation batch shape:\t", annotations.shape)

    epochs = 10
    R_STATE = 0
    BS = 32
    INIT_LR = 1e-3

    net_w = 128
    net_h = 128

    # STEPS = len(train_X) // BS ##steps_per_epoch

    class_limit = 20

    # Preprocessing - Images

    X = resize_arr(img_preprocess(stack_exp(images)),
                   net_h, net_w)
    print(X.shape)

    # Preprocessing - Annotations

    y = resize_arr(ann_preprocess(annotations, class_limit),
                   net_h, net_w)
    print(y.shape)

    # Splitting data
    train_data, test_data = split_data(X, y)

    model_1_input, model_1 = make_model(img_shape=[net_w, net_h],
                                        num_classes=class_limit,
                                        num_exposures=3)

    model_1_hist = model_1.fit(x=train_data[0],
                               y=train_data[1],
                               validation_data=test_data,
                               epochs=epochs)

    return model_1_hist.history.key()


if __name__ == "__main__":
    main()
