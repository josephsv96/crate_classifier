from data_loader import load_json, load_npy
from preprocessing import img_preprocess, ann_preprocess
from preprocessing import resize_arr, split_data, stack_exp
from model import make_model


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

    EPOCHS = 10
    R_STATE = 0
    BS = 32
    INIT_LR = 1e-3

    input_width = 128
    input_height = 128

    # STEPS = len(train_X) // BS ##steps_per_epoch

    class_limit = 20

    # Preprocessing - Images

    X = resize_arr(img_preprocess(stack_exp(images)),
                   input_height, input_width)
    print(X.shape)

    # Preprocessing - Annotations

    y = resize_arr(ann_preprocess(annotations, class_limit),
                   input_height, input_width)
    print(y.shape)

    # Splitting data
    train_data, test_data = split_data(X, y)

    model_1_input, model_1 = make_model(img_shape=[input_width, input_height],
                                        num_classes=class_limit,
                                        num_exposures=3)

    model_1_hist = model_1.fit(x=train_data[0],
                               y=train_data[1],
                               validation_data=test_data,
                               epochs=EPOCHS)

    return model_1_hist.history.key()


if __name__ == "__main__":
    main()
