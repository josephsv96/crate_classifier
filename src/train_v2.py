from pathlib import Path
from data_loader import load_npy
from preprocessing import img_preprocess, ann_preprocess
from preprocessing import resize_arr, split_data
from preprocessing import stack_exp_v2, ann_preprocess_v2
from model_v2 import CrateNet


def main():
    # loading files
    images = load_npy(Path(
        "C:/Users/josep/Documents/work/crate_classifier/dataset/data_1/dataset/data_1_images_v1.npy"))
    annots = load_npy(Path(
        "C:/Users/josep/Documents/work/crate_classifier/dataset/data_1/dataset/data_1_labels_v1.npy"))

    # Constants
    net_h = 128
    net_w = 128
    class_limit = 20
    LR = 1e-2
    EPOCHS = 100

    # Preprocessing
    img_arr = stack_exp_v2(images)
    annot_arr = ann_preprocess_v2(annots)
    X = resize_arr(img_preprocess(img_arr), net_h, net_w)
    print("X shape:", X.shape)
    y = resize_arr(ann_preprocess(annot_arr, class_limit), net_h, net_w)
    print("y shape:", y.shape)

    # Splitting data
    train_data, test_data = split_data(X, y)

    # Building and fitting model
    model = CrateNet.build(grid_h=net_h, grid_w=net_w, num_exp=3,
                           num_classes=class_limit, init_lr=LR, epochs=EPOCHS)
    model_hist = model.fit(x=train_data[0], y=train_data[1],
                           validation_data=test_data,
                           epochs=EPOCHS)
    return model_hist


if __name__ == "__main__":
    model_hist = main()
