# PLOTTING FRAMEWORK containing all visualizations presented in the final report

import os
import sys
import statistics
import numpy as np
import seaborn as sns
from datetime import datetime
from matplotlib import colors
from matplotlib import cm
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
from sklearn.metrics import confusion_matrix

workdir = os.path.dirname(__file__)
sys.path.append(workdir)  # append path of project folder directory

# DEFINE PLOT CHARACTERISTICS
width = 3.487
height = width / 1.618
resolution = 500
accent_color = '#808000'

rc('font', **{'family': 'serif', 'sans-serif': ['Libertine']})

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=10)
plt.rc('legend', fontsize=8)
# TITLES are not needed due to image captions in latex report


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


def plot_confusion_matrix(model, x_test, y_test):

    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    # use model to do the prediction
    y_pred = model.predict(x_test)
    # compute confusion matrix
    cm = confusion_matrix(np.argmax(np.array(y_test), axis=1), np.argmax(y_pred, axis=1))
    # plot confusion matrix
    plt.figure(figsize=(width, height), dpi=resolution)
    sns.heatmap(cm, cmap=get_continuous_cmap(['#ffffff', accent_color]),
                annot=True, fmt='d', linewidths=1, linecolor='black',
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes)

    plt.xlabel('Actual label')
    plt.ylabel('Predicted label')

    plt.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    plt.savefig(f"{workdir}/analysis_plots/confusion_matrix_testset{time_of_analysis}.png")
    plt.show()


def plot_box_sentiment(arr_diff, mean_sentiments, data_label):

    mean_sentiment_score = statistics.mean(arr_diff)
    mean_sentiment_score_str = str('{:0.2f}'.format(mean_sentiment_score))

    fig = plt.figure(dpi=resolution, figsize=(height*2, width))
    ax_top = fig.add_subplot(211)      # left plot     AX_LEFT
    ax_bottom = fig.add_subplot(212)     # right plot    AX_RIGHT

    # AX_LEFT create violin plot
    parts = ax_top.violinplot(arr_diff, points=100, vert=False,
                               showmeans=False, showextrema=False, showmedians=False)
    # AX_LEFT color the violin plot body
    for pc in parts['bodies']:
        pc.set_facecolor("olive")
        pc.set_edgecolor('black')
        pc.set_alpha(0.2)

    meanprops = dict(linestyle='-', linewidth=2, color='black')
    medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
    buckets = np.random.uniform(low=0.97, high=1.03, size=(len(arr_diff),))

    ax_top.scatter(arr_diff, buckets, edgecolors='black', color="olive", alpha=0.7, label=data_label)
    ax_top.boxplot(arr_diff, medianprops=medianprops, meanprops=meanprops,
                    showmeans=True, meanline=True, vert=False, showfliers=False)

    # AX_LEFT plot lines for mean indicator and plot separation
    ax_top.plot([mean_sentiment_score, mean_sentiment_score], [0.7, 0.75], color="olive", linewidth=3)
    ax_top.plot([mean_sentiment_score, mean_sentiment_score], [0.65, 1.3], color="olive", linewidth=1)
    ax_top.plot([-1.0, 1.0], [1.3, 1.3], color="grey", linewidth=1)

    # AX_LEFT add text label with information regarding Mean Sentiment
    ax_top.text(-1.028, 1.338, f' {mean_sentiment_score_str} mean sentiment ', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='olive', boxstyle='round'))

    # AX_LEFT add text labels with annotation for sentiment
    ax_top.text(-0.92, 0.73, 'negative', fontsize=8, weight='bold')
    ax_top.text(0.6, 0.73, 'positive', fontsize=8, weight='bold')

    # AX_LEFT titles, axis formatting, output
    #ax_top.set_title(f"Twitter Sentiment Analysis", pad=15, weight='bold')
    ax_top.set_xlabel("tweet sentiments", fontsize=10, weight='bold')
    ax_top.set_xlim(-1.1, 1.1)
    ax_top.set_ylim(0.7, 1.6)
    ax_top.yaxis.set_ticks([])
    ax_top.yaxis.set_ticklabels([])
    ax_top.grid(linestyle='--')
    ax_top.xaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    ax_top.legend(loc='upper left', prop={'size': 9})

    # AX_RIGHT configure inputs
    height_bar = [mean_sentiments[0],
                  mean_sentiments[1],
                  mean_sentiments[2]]
    bars = ('pos', 'neg', 'ntr')
    x_pos = np.arange(len(bars))

    # AX_RIGHT Create bar plot for the mean sentiments
    rects = ax_bottom.bar(x_pos, height_bar, color=['olive', 'grey', 'lightgrey'], edgecolor="black")
    ax_bottom.set_xticks(x_pos)              # define custom ticks
    ax_bottom.set_xticklabels(bars)          # name custom tick labels
    ax_bottom.bar_label(rects, padding=3)    # add label on top of bar

    ax_bottom.plot([-0.5, 2.5], [0.0, 0.0], color="black", linewidth=1)
    ax_bottom.plot([-0.5, 2.5], [1.0, 1.0], color="grey", linewidth=1)

    ax_bottom.set_xlabel("tweet sentiment classes", fontsize=10, weight='bold')
    ax_bottom.set_ylabel("mean sentiment distribution", fontsize=10, weight='bold')
    ax_bottom.yaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    ax_bottom.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_bottom.set_ylim(-0.05, 1.4)
    ax_bottom.grid(linestyle='--')

    # AX_RIGHT add labels for the bars
    colors_plot = {'positive  sentiment proportion': 'olive',
                   'negative sentiment proportion': 'grey',
                   'neutral   sentiment proportion': 'lightgrey'}
    labels = list(colors_plot.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors_plot[label]) for label in labels]
    ax_bottom.legend(handles, labels, loc='upper left', prop={'size': 9})

    # PLOT final output
    fig.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/analysis_plots/average_twitter_sentiment_{time_of_analysis}.png")
    fig.show()

def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = list(zip(y_data, x_data))
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height) 
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

def text_plotter(x_data, y_data, text, text_positions, axis,txt_width,txt_height):
    for x,y,text,t in zip(x_data, y_data, text, text_positions):
        axis.text(x - txt_width, 1.01*t, text,rotation=0, color='black', fontsize = 8)
        if y != t:
            axis.arrow(x, t,0,y-t, color='red',alpha=0.5, width=txt_width*0.00001, 
                       head_width=txt_width*0.0001, head_length=txt_height*0.0001, 
                       zorder=0,length_includes_head=True)

def plot_embeddings(embed_2d, label_array, indices_list):
    fig = plt.figure(figsize=(width, height), dpi=resolution)
    ax = fig.add_subplot()
    # scatter all points/embedding vectors
    ax.scatter(embed_2d[:, 0], embed_2d[:, 1], s = 1, c = '#808000', alpha = 0.1)

    # scatter similar embedding vectors
    # the analyzed word is shown in red and close-by vectors have the same color
    cmap = cm.get_cmap("tab10")
    for major_idx, indices in enumerate(indices_list):
        ax.scatter(embed_2d[indices, 0], embed_2d[indices, 1], s = 2, color = cmap(major_idx/len(indices_list)), alpha = 0.8)
        # mark word that the respective distance ranking is computed on
        ax.scatter(embed_2d[indices[0], 0], embed_2d[indices[0], 1], s = 4, c = "red")

    #set the bbox for the text. Increase txt_width for wider text.
    txt_height = 0.06*(plt.ylim()[1] - plt.ylim()[0])
    txt_width = 0.06*(plt.xlim()[1] - plt.xlim()[0])
    #Get the corrected text positions, then write the text.
    flat_indices = [index for indices in indices_list for index in indices]
    text_positions = get_text_positions(embed_2d[flat_indices, 0], embed_2d[flat_indices, 1], txt_width, txt_height)
    text_plotter(embed_2d[flat_indices, 0], embed_2d[flat_indices, 1], label_array[flat_indices], text_positions, ax, txt_width, txt_height)
    #for index in indices:
    #    ax.annotate(label_array[index], (embed_2d[index, 0], embed_2d[index, 1]), fontsize = 15)

    ax.set_xticks([])
    ax.set_yticks([])
    
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/analysis_plots/embedding_projection{time_of_analysis}.png")

    plt.tight_layout()
    plt.show()