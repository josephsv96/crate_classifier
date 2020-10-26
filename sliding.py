import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, annot_arr, w_scale):
        self.annot_arr = annot_arr
        self.N_dim = annot_arr.shape[0]
        self.w_scale, self.w_iter = self.get_params(self.N_dim, w_scale)
        self.w_dim = int(self.N_dim / self.w_scale)
        self.w_stride = int(self.w_dim / 2)

        self.sliced_arr = np.zeros((self.w_iter, self.w_dim, self.w_dim,
                                    annot_arr.shape[-1]))

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
        for j in range(self.xy_iter):
            y_1 = 0
            for i in range(self.xy_iter):
                x_2 = x_1 + self.w_dim
                y_2 = y_1 + self.w_dim

                # print([x_1, x_2], [y_1, y_2])
                self.sliced_arr[sec_no, :, :, :] = self.annot_arr[x_1:x_2,
                                                                  y_1:y_2, :]
                y_1 += self.w_stride
                sec_no += 1

            x_1 += self.w_stride

    def show_slices(self, num_class=10):
        """
        Prints all slices generated using the Sliding window
        """
        self.Slide()
        plt.figure(figsize=(10, 10))
        for im_num in range(self.w_iter):
            plt.subplot(self.xy_iter, self.xy_iter, im_num+1)
            if self.sliced_arr.shape[-1] == 1:
                plt.imshow(self.sliced_arr[im_num, :, :, 0])
            else:
                plt.imshow(self.sliced_arr[im_num, :, :, :3])
            plt.clim([0, num_class])
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()
