import numpy as np
import matplotlib.pyplot as plt

from segmap_v2 import SegmentaionMap
from utilities import get_custom_cmap, bgr_to_rgb_img


class SlidingWindow:
    """
    docstring
    """
    @staticmethod
    def get_params(N_dim, w_scale):
        """
        Returns a feasible sliding window scale and number of iterations
        """
        w_limit = int(np.log(N_dim)/np.log(2))
        # Possible scales
        w_scale_list = np.arange(1, w_limit + 1)
        # Possible iterations
        w_iter_list = np.arange(1, w_limit * 2, 2) ** 2

        # Checking validity of input scale
        if w_scale > w_limit or w_scale < 1:
            print("w_scale should be non-zero and meet the required condition")
            print(f"Condition: 1 <= w_scale <= {w_limit} for N_dim = {N_dim}")
            print(f"Using default, w_scale: {w_scale_list[int(w_limit/2)]}")
            w_scale = w_scale_list[int(w_limit/2)]

        w_iter = w_iter_list[np.where(w_scale_list == w_scale)[0][0]]

        # Missed pixels
        uncovered_px = N_dim % w_scale
        if uncovered_px != 0:
            print(f"Sliding window not able to capture {uncovered_px}px")

        # # DEBUG
        # print(f"w_scale_list: {w_scale_list}")
        # print(f"w_iter_list: {w_iter_list}")
        # print(f"w_scale: {w_scale}")
        # print(f"w_iter: {w_iter}")

        return w_scale, w_iter

    def __init__(self, annot_arr, w_scale, PARAMS):
        self.annot_arr = annot_arr
        self.N_dim = annot_arr.shape[0]
        self.w_scale, self.w_iter = self.get_params(self.N_dim, w_scale)
        self.w_dim = int(self.N_dim / self.w_scale)
        self.w_stride = int(self.w_dim / 2)

        self.sliced_arr = np.zeros((self.w_iter, self.w_dim, self.w_dim,
                                    annot_arr.shape[-1]))

        self.window_indices_x = []
        self.window_indices_y = []

        self.PARAMS = PARAMS

        # DEBUG
        # print(f"annot_arr.shape: {annot_arr.shape}")
        # print(f"N_dim: {self.N_dim}")
        # print(f"w_dim: {self.w_dim}")
        # print(f"w_stride: {self.w_stride}")
        # print(f"sliced_arr.shape: {self.sliced_arr.shape}")

    def Slide(self):
        """
        Moves a sliding window over the annot_arr of size w_dim
        Slices are updated into sliced_arr
        """
        self.xy_iter = int(np.power(self.w_iter, 0.5))

        x_1 = 0
        sec_no = 0  # Sector number (max => w_iter)
        for _ in range(self.xy_iter):
            y_1 = 0
            for _ in range(self.xy_iter):
                x_2 = x_1 + self.w_dim
                y_2 = y_1 + self.w_dim

                # print([x_1, x_2], [y_1, y_2])
                self.sliced_arr[sec_no, :, :, :] = self.annot_arr[x_1:x_2,
                                                                  y_1:y_2, :]
                self.window_indices_x.append(int(x_2 - self.w_dim/2))
                self.window_indices_y.append(int(y_2 - self.w_dim/2))

                y_1 += self.w_stride
                sec_no += 1

            x_1 += self.w_stride

    def show_slices(self, num_class=10):
        """
        Prints all slices generated using the Sliding window
        """
        CMAP = get_custom_cmap()
        self.Slide()
        plt.figure(figsize=(10, 10))
        for im_num in range(self.w_iter):
            plt.subplot(self.xy_iter, self.xy_iter, im_num+1)
            if self.sliced_arr.shape[-1] == 1:
                plt.imshow(self.sliced_arr[im_num, :, :, 0], cmap=CMAP)
            else:
                plt.imshow(self.sliced_arr[im_num, :, :, :3], cmap=CMAP)
            plt.clim([0, num_class])
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()

    def sort_img_arr_by_class(self):
        ann_arr = self.sliced_arr
        sector_name = [f"sec_{i}" for i in range(ann_arr.shape[0])]
        sec_db = {class_name: [] for class_name in sector_name}

        for i in range(ann_arr.shape[0]):
            annot = ann_arr[i, :, :, 0]
            segmap_obj = SegmentaionMap(annot,
                                        self.PARAMS["num_classes"],
                                        self.PARAMS["bg_class_id"])
            # fg_cls / activated_px, bg_cls/ total_px
            cls_den_arr = segmap_obj.activation_density()
            try:
                detected_cls = np.where(
                    cls_den_arr > self.PARAMS["det_thres"])[0][0]
            except IndexError:
                # Pushing images with less than threshold to or first id or last id
                detected_cls = 0

            sec_db[f"sec_{i}"].append(detected_cls)

        return list(sec_db.values())

    def show_slices_overlay(self, num_class=10):
        """
        Prints all slices generated using the Sliding window
        """
        sector_class = self.sort_img_arr_by_class()
        CMAP = get_custom_cmap()
        self.Slide()
        plt.figure(figsize=(10, 10))
        for im_num in range(self.w_iter):
            plt.subplot(self.xy_iter, self.xy_iter, im_num+1)
            if self.sliced_arr.shape[-1] == 1:
                plt.imshow(self.sliced_arr[im_num, :, :, 0])
                plt.clim([0, num_class])
                plt.scatter([int(self.w_dim/2)], [int(self.w_dim/2)],
                            s=100,
                            color=CMAP(sector_class[im_num]),
                            marker="s")
            else:
                plt.imshow(self.sliced_arr[im_num, :, :, :3])
                plt.clim([0, num_class])
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()

    def image_overlay(self, image_inst):
        sector_class = self.sort_img_arr_by_class()
        CMAP = get_custom_cmap()
        self.Slide()
        plt.figure(figsize=(10, 10))
        plt.imshow(bgr_to_rgb_img(image_inst))
        for im_num in range(self.w_iter):
            plt.scatter(self.window_indices_x[im_num],
                        self.window_indices_y[im_num],
                        s=100,
                        color=CMAP(sector_class[im_num]),
                        marker="s")
        plt.show()
