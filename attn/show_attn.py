import matplotlib
from matplotlib import pyplot as plt 
import numpy as np
# from show_attn_data_0 import X, S, S_next

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, cmap=plt.get_cmap("Purples"), vmin=0, vmax=0.7, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    # cbar = None

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=16)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=16)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="default")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def show_plots(x, s, s_next):
    i = 0
    print(len(s), len(s_next))
    for i in range(len(s)):
        print('i', i)
        s_i = s[i]
        s_next_i = s_next[i]
        show_plot(x, s_i, s_next_i)
        

def show_plot(x, s, s_next):
    # Prepare the data
    x = np.array(x)[0]
    s = np.array(s)[0]
    s_next = np.array(s_next)[0]
    nonzero = x.nonzero()[0].shape[0]
    x = x[:nonzero]
    s = s[:nonzero, :nonzero]
    s_next = s_next[:nonzero, :nonzero]

    x = [x_i-2 for x_i in x]
    x = ['?' if x_i < 0 else x_i for x_i in x]

    # Make the plot
    fig, ax = plt.subplots()
    fig_n, ax_n = plt.subplots()
    im, cbar = heatmap(s, x, x, ax=ax)
    plt.savefig('attn_s.png')
    im, cbar = heatmap(s_next, x, x, ax=ax_n)
    plt.savefig('attn_s_next.png')
    # im_n, cbar_n = heatmap(S_next[0], X[0], X[0], ax=ax)
    # texts = annotate_heatmap(im, valfmt="{x:.1f}")
    # fig.tight_layout()
    plt.show()



if __name__ == '__main__':


    from show_attn_data_3_long_3 import X, S, S_next
    # for data in ['show_attn_data_0', 'show_attn_data_1', 'show_attn_data_2']:
    #     from data 
    show_plots(X, S, S_next)

    # plt.save('mat.png')
    
    # plt.matshow(S[0])
    # plt.matshow(S_next[0])

    # plt.show()

    # plt.save('mat.png')
