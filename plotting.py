import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from wordcloud import WordCloud


def plot_single_cat(series, drop_nans=False, custom_order=None, horizontally=False,
                    ticks_rotation=0, save_path=None):
    """
    Plot bar chart of a single answer categorical value.
    :return:
    """
    val_counts = series.value_counts(sort=False, dropna=drop_nans)

    unique_answer_labels = val_counts.index.tolist()
    if not drop_nans:
        unique_answer_labels = [str(ele) for ele in unique_answer_labels]
    nbr_answered = val_counts.values.tolist()

    if custom_order:
        assert len(custom_order) == len(unique_answer_labels)

        zipped = zip(unique_answer_labels, nbr_answered, custom_order)
        sorted_vals = sorted(zipped, key=lambda x: x[2])
        unique_answer_labels = [ele[0] for ele in sorted_vals]
        nbr_answered = [ele[1] for ele in sorted_vals]

    labels = [f"{round(100*x/sum(nbr_answered),1)}%" for x in nbr_answered]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if horizontally:
        ax.barh(unique_answer_labels, nbr_answered)
        if custom_order:
            # barh normally goes from bottom to top, thus invert it if custom order is set.
            plt.gca().invert_yaxis()

        # Add value labels on next to the bar.
        rects = ax.patches
        for rect, label in zip(rects, labels):
            width = rect.get_width()
            ax.text(width + 0.3, rect.get_y() + rect.get_height() / 2, label, ha="left", va="center")
        # Increase x-max so that text doesn't overlap
        ax.set_xlim(xmax=ax.get_xlim()[1]*1.1)

    else:
        ax.bar(unique_answer_labels, nbr_answered)
        ax.set_xticklabels(unique_answer_labels, rotation=ticks_rotation)
        plt.ylabel("Times answered")

        # Add value labels on top of the bar.
        rects = ax.patches
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 0.3, label, ha="center", va="bottom")
        # Increase y-max so that text doesn't overlap
        ax.set_ylim(ymax=ax.get_ylim()[1] * 1.05)

    ax.grid(True)
    ax.set_axisbelow(True)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


def plot_multi_cat(labels, counts, total_answers, _sorted=True, ticks_rotation=0, save_path=None, figsize=(10, 5)):

    if _sorted:
        vals = zip(labels, counts)
        sorted_vals = sorted(vals, key=lambda x: x[1], reverse=True)
        labels = [v[0] for v in sorted_vals]
        counts = [v[1] for v in sorted_vals]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # ax.barh(labels, counts)
    ax.bar(labels, counts)
    ax.set_xticklabels(labels, rotation=ticks_rotation)

    percent_labels = [f"{round(100 * x / total_answers, 1)}%" for x in counts]
    # Add value labels on top of the bar.
    rects = ax.patches
    for rect, label in zip(rects, percent_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.3, label, ha="center", va="bottom")
    # Increase y-max so that text doesn't overlap
    ax.set_ylim(ymax=ax.get_ylim()[1] * 1.05)

    ax.grid(True)
    ax.set_axisbelow(True)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


def plot_wordcloud(text, save_path=None, **kwargs):
    """
    width=400, height=200, margin=0, max_words=200, min_font_size=4, max_font_size=None
    :param text:
    :param translate:
    :param kwargs:
    :return:
    """
    stop_words = set(stopwords.words('english'))

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    wordcloud = WordCloud(background_color='white', stopwords=stop_words, **kwargs).generate(text)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)



