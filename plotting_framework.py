# PLOTTING FRAMEWORK containing all visualizations presented in the final report

import os
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
from datetime import datetime
from sklearn.metrics import confusion_matrix

workdir = os.path.dirname(__file__)
sys.path.append(workdir)  # append path of project folder directory

# DEFINE PLOT CHARACTERISTICS
width = 3.487
height = width / 1.618
resolution = 500
accent_color = '#808000'

# pd.options.plotting.backend = "plotly"  # TODO wtf is this?

def get_continuous_cmap(hex_list, float_list=None):
    # creates and returns a color map that can be used in heat map figures.

    def hex_to_rgb(value):
        # Converts hex to rgb colours
        value = value.strip("#")  # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def rgb_to_dec(value):
        # Converts rgb to decimal colours (i.e. divides each value by 256)
        return [v / 256 for v in value]

    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    color_dict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        color_dict[col] = col_list
    cmp = colors.LinearSegmentedColormap('my_cmp', segmentdata=color_dict, N=256)
    return cmp


def plot_training_hist(history_import, epochs_count):

    """rc('font', **{'family': 'serif', 'sans-serif': ['Libertine']})
    rc('text', usetex=True)"""

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=10)
    plt.rc('figure', titlesize=8)
    plt.rc('legend', fontsize=8)

    fig, ax = plt.subplots(2, 1, figsize=(width, height*2), dpi=resolution)

    # Top plot: model ACCURACY
    ax[0].plot(history_import.history['accuracy'], color="darkslategrey")
    ax[0].plot(history_import.history['val_accuracy'], color=accent_color)
    # ax[0].set_xlabel('completed training epochs')
    ax[0].set_ylabel('model accuracy')
    ax[0].set_ylim(0.5, 1)
    ticks_loc = ax[0].get_xticks().tolist()
    ticks_loc = [int(x) for x in ticks_loc if x.is_integer()]
    ax[0].set_xticks(ticks_loc)
    ax[0].set_xticklabels([str(x+1) for x in ticks_loc])
    ax[0].set_xlim([-1, epochs_count])
    ax[0].legend(['Training', 'Validation'], loc='best')
    ax[0].grid(color='lightgrey', linestyle='--', linewidth=1)

    # Bottom plot: model LOSS
    ax[1].plot(history_import.history['loss'], color="darkslategrey")
    ax[1].plot(history_import.history['val_loss'], color=accent_color)
    ax[1].set_xlabel('completed training epochs')
    ax[1].set_ylabel('model loss')
    ax[1].set_ylim(0, 1)
    ticks_loc = ax[1].get_xticks().tolist()
    ticks_loc = [int(x) for x in ticks_loc if x.is_integer()]
    ax[1].set_xticks(ticks_loc)
    ax[1].set_xticklabels([str(x+1) for x in ticks_loc])
    ax[1].set_xlim([-1, epochs_count])
    ax[1].legend(['Training', 'Validation'], loc='best')
    ax[1].grid(color='lightgrey', linestyle='--', linewidth=1)

    fig.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/analysis_plots/accuracy_loss_{time_of_analysis}.png")
    fig.show()


def plot_confusion_matrix(model, X_test, y_test):

    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    # use model to do the prediction
    y_pred = model.predict(X_test)
    # compute confusion matrix
    cm = confusion_matrix(np.argmax(np.array(y_test), axis=1), np.argmax(y_pred, axis=1))
    # plot confusion matrix
    plt.figure(figsize=(width, height), dpi=resolution)
    sns.heatmap(cm, cmap=get_continuous_cmap(['#ffffff', accent_color, accent_color]),
                annot=True, fmt='d', linewidths=1, linecolor='black',
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes)

    plt.xlabel('Actual label')
    plt.ylabel('Predicted label')

    plt.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    plt.savefig(f"{workdir}/analysis_plots/confusion_matrix_testset{time_of_analysis}.png")
    plt.show()
