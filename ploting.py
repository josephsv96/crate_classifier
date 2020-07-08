import matplotlib.pyplot as plt


def lowest_point(arr):
    minimun_val = arr.index(min(arr))
    return minimun_val


def highest_point(arr):
    minimun_val = arr.index(max(arr))
    return minimun_val


def plot_history(history_list, epochs=EPOCHS):
    plt.figure(figsize=(30, 15))
    plot_colors = ['b', 'm', 'r', 'g']
    #  "Accuracies"
    plt.subplot(1, 2, 1)
    plot_legend = []
    print("Max Accuracies:")
    for i, model_history in enumerate(history_list):
        print("(Acc) model_" + str(i+1) + ":",
              max(model_history.history['accuracy']) * 100,
              "@", highest_point(model_history.history['accuracy']),
              "/"+str(epochs))
        print("(Val) model_" + str(i+1) + ":",
              max(model_history.history['val_accuracy']) * 100,
              "@", highest_point(model_history.history['val_accuracy']),
              "/"+str(epochs))
        print("\t")
        # Plots
        plt.plot(model_history.history['accuracy'], '--' + plot_colors[i])
        plt.plot(model_history.history['val_accuracy'], plot_colors[i])
        # plt.hlines(y=max(model_history.history['accuracy']), xmin=0, xmax=epochs)
        plot_legend.append('model_' + str(i+1) + '_train')
        plot_legend.append('model_' + str(i+1) + '_val')
    # plot labels
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(plot_legend, loc='best')

    # "Losses"
    plt.subplot(1, 2, 2)
    plot_legend = []
    for i, model_history in enumerate(history_list):
        plt.plot(model_history.history['loss'], '--' + plot_colors[i])
        plt.plot(model_history.history['val_loss'], plot_colors[i])
        # plt.hlines(y=min(model_history.history['loss']), xmin=0, xmax=epochs)
        plot_legend.append('model_' + str(i+1) + '_train')
        plot_legend.append('model_' + str(i+1) + '_val')
    # plot labels
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(plot_legend, loc='best')

    plt.show()


# For plotting from list of model histories with, element shape [acc, val_acc]
def plot_history_v2(history_list, class_colors):
    plt.figure(figsize=(30, 15))
    plot_colors = class_colors
    #  "Accuracies"
    plt.subplot(1, 2, 1)
    plot_legend = []
    print("Max Accuracies:")
    for i, model_history in enumerate(history_list):
        print("(Acc) model_" + str(i+1) + ":",
              max(model_history[0].history['accuracy']) * 100,
              "@", highest_point(model_history[0].history['accuracy']),
              "/" + str(EPOCHS))
        print("(Val) model_" + str(i+1) + ":",
              max(model_history[1].history['accuracy']) * 100,
              "@", highest_point(model_history[1].history['val_accuracy']),
              "/" + str(EPOCHS))
        print("\t")
        # Plots
        plt.plot(model_history[0].history['accuracy'],
                 linestyle='--', color=plot_colors[i])
        plt.plot(model_history[1].history['val_accuracy'],
                 color=plot_colors[i])
        plot_legend.append('model_' + str(i+1) + '_train')
        plot_legend.append('model_' + str(i+1) + '_val')
    # plot labels
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(plot_legend, loc='best')

    # "Losses"
    plt.subplot(1, 2, 2)
    plot_legend = []
    for i, model_history in enumerate(history_list):
        plt.plot(model_history[0].history['loss'],
                 linestyle='--', color=plot_colors[i])
        plt.plot(model_history[1].history['val_loss'], color=plot_colors[i])
        plot_legend.append('model_' + str(i+1) + '_train')
        plot_legend.append('model_' + str(i+1) + '_val')
    # plot labels
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(plot_legend, loc='best')

    plt.show()


# For plotting from list of model histories with, element shape [acc, val_acc]
def plot_history_v3(history_list, class_colors):
    plt.figure(figsize=(30, 15))
    plot_colors = class_colors
    #  "Accuracies"
    plt.subplot(1, 2, 1)
    plot_legend = []
    print("Max Accuracies:")
    for i, model_history in enumerate(history_list):
        # print("(Acc) model_" + str(i+1) + ":",
        #       max(model_history[0].history['accuracy'])* 100 ,
        #       "@",highest_point(model_history[0].history['accuracy']),
        #       "/" + str(EPOCHS))
        print("(Val) model_" + str(i+1) + ":",
              max(model_history[1].history['accuracy']) * 100,
              "@", highest_point(model_history[1].history['val_accuracy']),
              "/" + str(EPOCHS))
        print("\t")
        # Plots
        # plt.plot(model_history[0].history['accuracy'],
        # linestyle = '--', color=plot_colors[i])
        plt.plot(model_history[1].history['val_accuracy'],
                 color=plot_colors[i])
        #plot_legend.append('model_' + str(i+1) + '_train')
        plot_legend.append('model_' + str(i+1) + '_val')
    # plot labels
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(plot_legend, loc='best')

    # "Losses"
    plt.subplot(1, 2, 2)
    plot_legend = []
    for i, model_history in enumerate(history_list):
        # plt.plot(model_history[0].history['loss'],
        # linestyle = '--', color=plot_colors[i])
        plt.plot(model_history[1].history['val_loss'], color=plot_colors[i])
        # plot_legend.append('model_' + str(i+1) + '_train')
        plot_legend.append('model_' + str(i+1) + '_val')
    # plot labels
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(plot_legend, loc='best')

    plt.show()

# For plotting from list of model histories with, multiple outputs


def plot_history_v4(history_list, epochs=EPOCHS):
    plt.figure(figsize=(30, 15))
    plot_colors = ['b', 'm', 'r', 'g']
    #  "Accuracies"
    plt.subplot(1, 2, 1)
    plot_legend = []
    print("Max Accuracies:")
    for i, model_history in enumerate(history_list):
        print("(Acc) model_" + str(i+1) + ":",
              max(model_history.history['category_accuracy']) * 100,
              "@", highest_point(model_history.history['category_accuracy']),
              "/"+str(epochs))
        print("(Acc) model_" + str(i+1) + ":",
              max(model_history.history['isCrate_accuracy']) * 100,
              "@", highest_point(model_history.history['isCrate_accuracy']),
              "/"+str(epochs))
        print("\t")
        # Plots
        plt.plot(model_history.history['category_accuracy'], plot_colors[i])
        plt.plot(
            model_history.history['isCrate_accuracy'], '--' + plot_colors[i])
        plt.plot(
            model_history.history['val_category_accuracy'], plot_colors[i+1])
        plt.plot(
            model_history.history['val_isCrate_accuracy'], '--' + plot_colors[i+1])
        # plt.hlines(y=max(model_history.history['category_accuracy']),
        # xmin=0, xmax=epochs)
        plot_legend.append('model_' + str(i+1) + '_train')
        plot_legend.append('model_' + str(i+1) + '_val')
    # plot labels
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(plot_legend, loc='best')

    # "Losses"
    plt.subplot(1, 2, 2)
    plot_legend = []
    for i, model_history in enumerate(history_list):
        plt.plot(model_history.history['category_loss'], '--' + plot_colors[i])
        plt.plot(model_history.history['isCrate_loss'], '--' + plot_colors[i])
        plt.plot(
            model_history.history['val_category_loss'], '--' + plot_colors[i+1])
        plt.plot(
            model_history.history['val_isCrate_loss'], '--' + plot_colors[i+1])
        # plt.hlines(y=min(model_history.history['loss']), xmin=0, xmax=epochs)
        plot_legend.append('model_' + str(i+1) + '_train')
        plot_legend.append('model_' + str(i+1) + '_val')
    # plot labels
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(plot_legend, loc='best')

    plt.show()
