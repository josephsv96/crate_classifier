import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from scipy.stats import norm


class SegmentaionMap:
    def __init__(self, annot_inst):
        self.annot_inst = np.expand_dims(annot_inst, -1)
        self.class_colors = ['black', '#259c14', '#4c87c6',
                             '#737373', '#cbec24', '#f0441a', '#0d218f']
        # 0 -> BG, 8 -> Sticker, 9 -> Bottle
        self.bg_class_index = [0, 8, 9]

    def arr_to_categorical(self):
        annot_cat = to_categorical(self.annot_inst,
                                   num_classes=None, dtype='float32')

        return annot_cat

    def activation_density(self):
        image_arr = self.arr_to_categorical()

        num_classes = image_arr.shape[-1]
        activation_densities = np.zeros(num_classes)
        max_pixels = image_arr.shape[0] * image_arr.shape[1]
        total_pixels = image_arr.shape[0] * image_arr.shape[1]

        # Reducing effective area of activations
        for i in self.bg_class_index:
            data = image_arr[:, :, i]
            total_pixels -= data[np.where(data > 0)].shape[0]

        # Densitites of activated classes wrt effective area
        for i in range(num_classes):
            image = image_arr[:, :, i]
            activated_pixels = image[image > 0]
            if total_pixels > 0:
                activation_densities[i] = activated_pixels.shape[0] / \
                    total_pixels
            else:
                activation_densities[i] = 0

        # Densities of bg classes wrt full slice
        for i in self.bg_class_index:
            image = image_arr[:, :, i]
            activated_pixels = image[image > 0]
            activation_densities[i] = activated_pixels.shape[0] / max_pixels

        return activation_densities

    def imshow_prediciton(self):
        densities_list = self.get_density()

        plt.figure(figsize=(15, 10))
        j = 0
        num_of_images = self.annot_inst.shape[0] + 1
        if num_of_images > 10:
            num_of_images = 10
        for i in range(self.annot_inst.shape[0]):
            plt.subplot(1 * num_of_images, 2, j+1)
            plt.imshow(self.annot_inst[i, :, :])
            # print(densities_list[i, 1:] * 100)

            plt.subplot(1 * num_of_images, 2, j+2)
            plt.plot(densities_list[i, :] * 100)
            plt.xlabel('Class')

            j = j + 2

    # PART 1
    def class_dist(self, low_threshold=0.15):
        densities_list = self.get_density()
        num_classes = densities_list.shape[-1]
        num_annots = densities_list.shape[0]

        # up_threshold = 0.45
        # Average of prediction over all classes
        class_dist = np.zeros([num_classes, num_annots, num_classes])
        for i in range(num_annots):
            class_index = np.where(densities_list[i, 1:] > low_threshold)[0][0]
            class_dist[class_index, i, 1:] = densities_list[i, 1:]

        class_dist = np.sum(class_dist, axis=1)

        return class_dist

    def show_class_dist(self):
        densities_list = self.class_dist()
        num_classes = densities_list.shape[0]
        class_colors = self.class_colors
        plt.figure(figsize=(12, 24))

        for i in range(num_classes):
            plt.subplot(1 * num_classes, 1, i+1)
            plt.plot(densities_list[i, :] * 100, color=class_colors[i])
            # print(densities_list[i, 1:] * 100)
            plt.legend(['Class_' + str(i)])
            plt.xlabel('Classes')
            plt.ylabel('Prediciton')
            plt.ylim(0)

    # PART 2
    def batch_dist(self, low_threshold=0.15, low_cutoff=0.05):
        densities_list = self.get_density()
        batch_dist = densities_list.transpose()
        num_classes = densities_list.shape[1]
        new_dist = []

        for i in range(num_classes):
            class_index = np.where(batch_dist[i, :] > low_threshold)[0]
            new_dist.append(densities_list[class_index, :])

        # To remove very low values that disturb the distribution
        for i in range(num_classes):
            sample_x = new_dist[i]
            for j in range(num_classes):
                x_values = sample_x[:, j]
                mean = np.mean(x_values)
                if mean < low_cutoff:
                    new_dist[i][:, j] = np.zeros(x_values.shape)
        # This needs to be improved
        return new_dist, batch_dist

    def get_norm_values(self, x_values, plot_type='a'):
        x_values_norm = 1
        if plot_type != 'a':
            x_values_norm = np.linalg.norm(x_values)
        x_values = np.sort(x_values / x_values_norm)
        mean = np.mean(x_values)
        std = np.std(x_values)
        y_values = norm.pdf(x_values, mean, std)

        return x_values, y_values, mean, std

    def show_batch_dist(self, show_bg=False, max_y_limit=10, plot_type='a'):
        new_dist, _ = self.batch_dist()
        num_classes = len(new_dist)
        if show_bg:
            class_list = np.arange(num_classes)
        else:
            class_list = np.arange(1, num_classes)

        class_colors = self.class_colors

        plt.figure(figsize=(10, 24))
        for i in class_list:
            plt.subplot(num_classes, 1, i+1)
            sample_x = new_dist[i]
            sample_y = np.zeros(sample_x.shape)
            legends = []
            for j in class_list:
                x_values, y_values, mean, std = self.get_norm_values(
                    sample_x[:, j], plot_type)

                if mean != 0:
                    legends.append('Class_' + str(j) +
                                   ", Mean: " + str(round(mean, 2)))
                    if plot_type == 'a':
                        plt.scatter(sample_x[:, j], sample_y[:, j],
                                    color=class_colors[j])

                    plt.plot(x_values, y_values, color=class_colors[j])
                    plt.axvline(mean, color=class_colors[j], linestyle='--')
                    plt.legend(legends)
            plt.xlabel('Frequency')
            plt.ylabel('PDF')
            plt.ylim(0, max_y_limit)
            plt.xlim(0, 1)
        plt.show()
