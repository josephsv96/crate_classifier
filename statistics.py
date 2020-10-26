# Importing modules
import numpy as np
import matplotlib.pyplot as plt


def class_identifier(annot_arr):
    """Identifies the "major" class found in the annotation

    Args:
        annot_arr ([type]): [description]

    Returns:
        [type]: [description]
    """
    class_list = []
    # loop to find unique class list
    for i in range(annot_arr.shape[0]):
        annot_item = annot_arr[i, :, :]
        flatten_arr = annot_item.astype(dtype="int64").flatten()
        frequent_class = np.argmax(np.bincount(flatten_arr))
        class_list.append(frequent_class)

    return np.array(class_list)


def show_class_dist(annot_arr):
    class_list = class_identifier(annot_arr)
    color_list = ["#288e08", "#1cd7e3", "#a4a4a4", "#d5f017",
                  "#c86110", "#094795", "#3b109f", "#5e089b"]
    x = [f"class_{i}" for i in range(1, np.max(class_list)+1)]
    y = np.bincount(class_list)[1:]

    x_pos = [i for i, _ in enumerate(x)]

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color=color_list)

    # Mean and stdard dev
    mu = np.mean(y)
    stdev = np.std(y)
    plt.hlines(y=mu, xmin=x[0], xmax=x[-1], label="mu")
    plt.hlines(y=mu + stdev, xmin=x[0], xmax=x[-1],
               linestyles="dotted", label="+sigma")
    plt.hlines(y=mu - stdev, xmin=x[0], xmax=x[-1],
               linestyles="dotted", label="-sigma")

    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title("Class Distribution")
    plt.xticks(x_pos, x)

    plt.show()