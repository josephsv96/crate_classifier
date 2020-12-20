import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

# Local modules

from data_loader import load_json, load_npy
from preprocessing import img_preprocess, ann_preprocess
from preprocessing import resize_arr, split_data, stack_exp
from preprocessing import stack_exp_v2, ann_preprocess_v2
from utilities import img_arr_to_gray
# import augmentation_dev as augmentation
# import plotting
# from train import train_model_1, train_model_2, train_model_1_v2

# Models

import model_14k as model_1    # baseline model
# VGG16 models
import model_vgg16_24k as model_2
import model_vgg16_34k as model_3
import model_vgg16_47k as model_4
# ResNet models
import model_resnet18_34k as model_5
import model_resnet18_46k as model_6
# DenseNet models
import model_densenet21_35k as model_7
import model_densenet21_48k as model_8


# Colormap object for custom colours
from matplotlib.colors import LinearSegmentedColormap

# Colors
class_colors = [(0, 0, 0),
                (0, 0.5, 0), (0, 1, 1), (0.5, 0.5, 0.5), (0.5, 1, 0.5),
                (0.5, 0.25, 0.25), (0.5, 0, 1), (0, 0.25, 0.5), (0, 0, 1),
                (1, 1, 1), (1, 0, 0.5)]

CMAP_11 = LinearSegmentedColormap.from_list("cmap_11", class_colors, N=11)


# Global
num_exp = 3
model_list = [model_1,
              model_2, model_3, model_4,
              model_5, model_6,
              model_7, model_8]
model_names = ["model_14k",
               "VGG16_24k", "VGG16_34k", "VGG16_47k",
               "ResNet18_34k", "ResNet18_46k",
               "DenseNet21_35k", "DenseNet21_48k"]


# Constants
net_h = 128
net_w = 128
class_limit = 20

R_STATE = 0
BS = 32
LR = 1e-2
PRE_EPOCHS = 500
EPOCHS = 500

# Converting from RGB to BGR


def bgr_to_rgb(images):
    buffer_arr = np.zeros((images.shape))
    j = 0
    for i in range(num_exp):
        buffer_arr[:, :, :, j] = images[:, :, :, j+2]
        buffer_arr[:, :, :, j+1] = images[:, :, :, j+1]
        buffer_arr[:, :, :, j+2] = images[:, :, :, j]
        j += 3

    return buffer_arr


# Preview of data

def preview_data_v2(img_arr, annot_arr, num_exp, index=0):
    plt.figure(figsize=(20, 50))
    # showing images
    j = 0
    for i in range(num_exp):
        plt.subplot(1, num_exp+2, i+1)
        plt.imshow(img_arr[index, :, :, j:j+num_exp]/255)
        plt.xlabel(f"exp_{i}")
        j += num_exp
        plt.xticks([])
        plt.yticks([])
    # showing annotations
    plt.subplot(1, num_exp+2, i+2)
    plt.imshow(annot_arr[index, :, :, 0])
    plt.xlabel("mask")
    plt.xticks([])
    plt.yticks([])
    # showing label
    plt.show()


# Misc

def print_all_preds(img_arr, y_test, y_pred, cols=5, class_to_show=6, color_mode='rgb'):

    num = y_test.shape[0]

    if (num % cols != 0):
        rows = int(num/cols) + 1
    else:
        rows = int(num/cols)

    fig_size = 2
    col_set = cols * 3
    k = 0
    # Creating a img grid
    plt.figure(figsize=(fig_size*col_set, fig_size*rows))

    plt_num = 1
    rgb_img = np.zeros([img_arr.shape[1], img_arr.shape[2], 3])
    for i in range(rows):
        for j in range(cols):
            if k == num:
                break
            if color_mode == 'bgr':
                rgb_img[:, :, 0] = img_arr[k, :, :, 2]
                rgb_img[:, :, 1] = img_arr[k, :, :, 1]
                rgb_img[:, :, 2] = img_arr[k, :, :, 0]
            else:
                rgb_img = img_arr[k, :, :, 0:3]

            plt.subplot(rows, col_set, plt_num)
            plt.imshow(rgb_img)
            plt.xlabel("Image")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(rows, col_set, plt_num+1)
            plt.imshow(y_test[k])
            plt.xlabel("Ground_Truth")
            plt.xticks([])
            plt.yticks([])
            plt.clim(vmin=0, vmax=class_to_show)

            plt.subplot(rows, col_set, plt_num+2)
            plt.imshow(y_pred[k])
            plt.xlabel("Prediction")
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.clim(vmin=0, vmax=class_to_show)

            plt_num = plt_num + 3
            k += 1
    plt.tight_layout()
    plt.show()
    # plt.savefig("/content/drive/My Drive/rhs_werk_March2020/dataset_3_ann/all_preds.png")


def print_all_preds_v2(img_arr, model_pred, cols=5, class_to_show=6, color_mode='rgb'):
    # arrays
    is_crate_arr = model_pred[:, :, :, 0]
    class_arr = np.argmax(model_pred[:, :, :, 1:], axis=-1)

    # Grid spec
    num = model_pred.shape[0]
    if (num % cols != 0):
        rows = int(num/cols) + 1
    else:
        rows = int(num/cols)
    fig_size = 2
    col_set = cols * 3
    k = 0
    # Creating a img grid
    plt.figure(figsize=(fig_size*col_set, fig_size*rows))

    plt_num = 1
    rgb_img = np.zeros([img_arr.shape[1], img_arr.shape[2], 3])
    for i in range(rows):
        for j in range(cols):
            if k == num:
                break
            if color_mode == 'bgr':
                rgb_img[:, :, 0] = img_arr[k, :, :, 2]
                rgb_img[:, :, 1] = img_arr[k, :, :, 1]
                rgb_img[:, :, 2] = img_arr[k, :, :, 0]
            else:
                rgb_img = img_arr[k, :, :, 0:3]

            plt.subplot(rows, col_set, plt_num)
            plt.imshow(rgb_img)
            plt.xlabel("Image")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(rows, col_set, plt_num+1)
            plt.imshow(is_crate_arr[k])
            plt.xlabel("is_Crate")
            plt.xticks([])
            plt.yticks([])
            plt.clim(vmin=0, vmax=1)

            plt.subplot(rows, col_set, plt_num+2)
            plt.imshow(class_arr[k, :, :])
            plt.xlabel("class")
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.clim(vmin=0, vmax=class_to_show)

            plt_num = plt_num + 3
            k += 1
    plt.tight_layout()
    plt.show()
    # plt.savefig("/content/drive/My Drive/rhs_werk_March2020/dataset_3_ann/all_preds.png")


def pred_to_img(y_test, y_pred):
    y_test = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    return y_test, y_pred


def show_model_pred(model, img_arr, annot_arr, net_h, net_w, class_to_show, color_mode):
    img_arr = resize_arr(img_preprocess(img_arr), net_h, net_w)
    model_pred = model.predict(img_arr)
    model_true = resize_arr(ann_preprocess(
        annot_arr, class_limit), net_h, net_w)
    y_test, y_pred = pred_to_img(model_true, model_pred)
    print_all_preds(img_arr, y_test, y_pred, cols=4,
                    class_to_show=class_to_show, color_mode=color_mode)

    return model_pred


def show_model_pred_2(model, img_arr, annot_arr, net_h, net_w, class_to_show, color_mode):
    img_arr = resize_arr(img_preprocess(img_arr), net_h, net_w)
    model_pred = model.predict(img_arr)
    # model_true = resize_arr(ann_preprocess(annot_arr,class_limit), net_h, net_w)
    print_all_preds_v2(img_arr, model_pred, cols=4,
                       class_to_show=class_to_show, color_mode=color_mode)

    return model_pred


def synth_exp(image_arr, num_exp=3):
    new_image_arr = np.zeros(
        [image_arr.shape[0], image_arr.shape[1], image_arr.shape[2], num_exp*3])
    for i in range(image_arr.shape[0]):
        new_image_arr[i, :, :, 0:3] = image_arr[i, :, :, :] - 70
        new_image_arr[i, :, :, 3:6] = image_arr[i, :, :, :]
        new_image_arr[i, :, :, 6:9] = image_arr[i, :, :, :] + 70
    new_image_arr[new_image_arr > 255] = 255.0
    new_image_arr[new_image_arr < 0] = 0.0
    return new_image_arr


def join_2_arr(arr_1, arr_2):
    img_num = arr_1.shape[0] + arr_2.shape[0]
    joined_arr = np.zeros(
        [img_num, arr_1.shape[1], arr_1.shape[2], arr_1.shape[-1]])

    joined_arr[0:arr_1.shape[0], :, :, :] = arr_1
    joined_arr[arr_1.shape[0]:img_num, :, :, :] = arr_2

    return joined_arr

# Plotting


def lowest_point(arr):
    minimun_val = arr.index(min(arr))
    return minimun_val


def highest_point(arr):
    minimun_val = arr.index(max(arr))
    return minimun_val


def plot_history_mode(history_list, epochs, models_list, model_names=None, plot_mode="train"):
    # Setting modes (train or val)
    if plot_mode == "train":
        acc_mode = "accuracy"
        loss_mode = "loss"
        style = "dashed"
    elif plot_mode == "test":
        acc_mode = "val_accuracy"
        loss_mode = "val_loss"
        style = "solid"
    else:
        print("ERROR: Invalid plot_mode")

    plt.figure(figsize=(30, 15))
    #plot_colors = ['b', 'm', 'r', 'g', ]
    # Initializing names
    if model_names is None or len(model_names) != len(history_list):
        model_names = [f"model_1{i}" for i in range(len(history_list))]
    #  "Accuracies"
    plt.subplot(1, 2, 1)
    plot_legend = []
    print("Max Accuracies:")
    for i, model_history in enumerate(history_list):
        max_acc_val = max(model_history.history[acc_mode]) * 100
        max_pt_val = highest_point(model_history.history[acc_mode])

        print(
            f"({plot_mode}) {model_names[i]}\t: {max_acc_val:.2f} @ {max_pt_val}/{epochs}")
        # Plots
        plt.plot(model_history.history[acc_mode], linestyle=style)
        # plt.hlines(y=max(model_history.history['accuracy']), xmin=0, xmax=epochs)
        plot_legend.append(
            f'{model_names[i]} [{models_list[i].count_params():,d}]')
    # plot labels
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(plot_legend, loc='best')

    # "Losses"
    plt.subplot(1, 2, 2)
    plot_legend = []
    for i, model_history in enumerate(history_list):
        plt.plot(model_history.history[loss_mode], linestyle=style)
        # plt.hlines(y=min(model_history.history['loss']), xmin=0, xmax=epochs)
        plot_legend.append(
            f'{model_names[i]} [{models_list[i].count_params():,d}]')
    # plot labels
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(plot_legend, loc='best')

    plt.show()

# Training Helper funcs

# Preprocessing for Model 1 ("isCrate")


def preprocess_iscrate(img_arr, annot_arr):
    X = resize_arr(img_preprocess(img_arr), net_h, net_w)
    # print("X shape:", X.shape)
    # convert y to "isCrate" classes
    y = resize_arr(ann_preprocess(
        annot_arr[:, :, :, 0], class_limit), net_h, net_w)
    y = y[:, :, :, 0:2]
    y[:, :, :, 1] = (y[:, :, :, 0] * - 1) + 1
    # print("y shape:", y.shape)

    # Splitting data to detect "isCrate"
    train_data, test_data = split_data(X, y)

    return (train_data, test_data)

# Preprocessing for Model 2 ("isClass")


def preprocess_isclass(img_arr, annot_arr):
    X = resize_arr(img_preprocess(img_arr), net_h, net_w)
    # print("X shape:", X.shape)
    y = resize_arr(ann_preprocess(
        annot_arr[:, :, :, 0], class_limit), net_h, net_w)
    # print("y shape:", y.shape)

    # Splitting data to detect "isCrate"
    train_data, test_data = split_data(X, y)

    return (train_data, test_data)


def bundle_models(model_list, height=net_h, width=net_w, exp_num=3, class_num=2, net_depth=3, LR=LR, EPOCHS=EPOCHS):
    """
    Returns a list of compiled models with given config and model defenitions
    """
    compiled_models = []
    for i, model_def in enumerate(model_list):
        print(f"Building model_{i+1}/{len(model_list)}")
        model_comp = model_def.CrateNet.build(grid_h=height, grid_w=width,
                                              num_exp=exp_num,
                                              num_classes=class_num,
                                              depth=net_depth,
                                              init_lr=LR, epochs=EPOCHS)
        compiled_models.append(model_comp)
    return compiled_models


def train_model_bundle(models, train_data, test_data, EPOCHS=EPOCHS, to_h5=True, mod_no='1'):
    """Training model from the list of defined models"""
    models_hist = []
    for i, model in enumerate(models):
        print(f"Training model_{i+1}/{len(models)}")
        models_hist.append(model.fit(x=train_data[0], y=train_data[1],
                                     validation_data=test_data, epochs=EPOCHS))
        if to_h5:
            model.save("mod_" + mod_no + '_' + model_names[i] + ".h5")
            model.save_weights("mod_" + mod_no + '_' +
                               model_names[i] + "_w.h5")
    return models_hist


# Crate detector

# Crate detector
def crate_detector(model_1, model_2, img_arr, annnot_arr, index):
    output_1 = np.argmax(model_1.predict(
        np.expand_dims(img_arr[index], axis=0)), axis=-1)[0, :, :]
    output_2 = np.argmax(model_2.predict(
        np.expand_dims(img_arr[index], axis=0)), axis=-1)[0, :, :]

    # Colors
    class_colors = ['black', '#259c14', '#4c87c6', '#737373', '#cbec24',
                    '#f0441a', '#0d218f', 'blue', 'magenta', 'green']

    output_3 = output_1 * output_2
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(img_arr[index, :, :, 6:9], cmap="cividis")
    plt.title("Image")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 5, 2)
    plt.imshow(np.argmax(annnot_arr[index, :, :, :], axis=-1), cmap="cividis")
    plt.title("Groud truth")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.clim(0, 10)

    plt.subplot(1, 5, 3)
    plt.imshow(output_1, cmap="cividis")
    plt.title("model_1 output")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.clim(0, 1)

    plt.subplot(1, 5, 4)
    plt.imshow(output_2, cmap="cividis")
    plt.title("model_2 output")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.clim(0, 10)

    plt.subplot(1, 5, 5)
    plt.imshow(output_3, cmap="cividis")
    plt.title("combined output")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.clim(0, 10)

    plt.show()
    return output_3


def load_mod_weights(model_list, model_names, prefix="mod_1_"):
    loaded_model_list = []
    for model, name in zip(model_list, model_names):
        loaded_model_list.append(model.load_weights(prefix + name + ".h5"))
        print(model)
        print(prefix + name + ".h5")
        print(loaded_model_list)
    return loaded_model_list


class ModelBundle:

    def __init__(self, model_list, model_names, net_config, train_config):
        self.model_list = model_list
        self.model_names = model_names

        self.height = net_config["net_h"]
        self.width = net_config["net_w"]
        self.num_exp = net_config["num_exp"]
        self.net_depth = net_config["net_depth"]

        self.LR = train_config["learning_rate"]
        self.EPOCHS = train_config["epochs"]

    def bundle_models(self, class_num=2):
        """
        Returns a list of compiled models with given config and model
        defenitions
        """
        compiled_models = []
        for i, model_def in enumerate(model_list):
            print(f"Building model_{i+1}/{len(model_list)}")
            model_comp = model_def.CrateNet.build(grid_h=self.height,
                                                  grid_w=self.width,
                                                  num_exp=self.num_exp,
                                                  num_classes=class_num,
                                                  depth=self.net_depth,
                                                  init_lr=self.LR,
                                                  epochs=self.EPOCHS)
            compiled_models.append(model_comp)

        return compiled_models

    def train_bundle(self, train_d, test_d, class_num, mod_no=1, to_h5=True):
        """Training model from the list of defined models"""

        c_models = self.bundle_models(class_num)
        models_hist = []

        for i, model in enumerate(c_models):
            print(f"Training (mod_{mod_no}) {self.model_names[i]}")
            print(f"{i+1}/{len(c_models)}")

            # Loop over the different datasets
            models_hist.append(model.fit(x=train_d[0], y=train_d[1],
                                         validation_data=test_d,
                                         epochs=EPOCHS))
            if to_h5:
                model_name_i = "mod_" + str(mod_no) + '_' + model_names[i]
                print(f"Saving model: {model_name_i}")
                model.save(model_name_i + ".h5")
                model.save_weights(model_name_i + "_w.h5")

        return models_hist
