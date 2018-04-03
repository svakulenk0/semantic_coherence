# -*- coding: utf-8 -*-
'''
svakulenko
29 Mar 2018

Generate plots of distributions from dictionary or Counter objects
'''
import matplotlib.pyplot as plt

from sample291848 import word_distribution, entity_distribution, get_top
from prepare_dataset import VOCAB_WORDS_PATH, VOCAB_ENTITIES_PATH
# from annotate_shortest_paths import SAMPLE_4606


def make_boxplot(data, label='new'):
    plt.boxplot(data)
    plt.show()


def plot_histogram(x, y, size=(30, 5), label='new', xlabels=None, flip=False):
    '''
    from https://stackoverflow.com/questions/21195179/plot-a-histogram-from-a-dictionary
    '''
    print y
    plt.figure(figsize=(30,5))
    if x == None:
        x = range(len(y))
    width = 1.0  # gives histogram aspect to the bar diagram
    
    # horizontal bar plot
    if flip:
        plt.barh(x, y, width, color='g')
        if xlabels:
            plt.yticks(x, xlabels)
    else:
        plt.bar(x, y, width, color='g')
        if xlabels:
            plt.xticks(x, xlabels)  # , rotation='vertical'
    plt.show()
    # plt.savefig('distr_%s.pdf'%label)


if __name__ == '__main__':
    # plot_histogram(n_entities_per_dialogue.keys(), n_entities_per_dialogue.values(), label='n_entities_per_dialogue')
    n = 5
    
    
    plot_histogram(None, entity_distribution.most_common()[:n], (15, 30), label='entity_distribution', xlabels=get_top(entity_distribution, VOCAB_ENTITIES_PATH, n))
        # coherence_4606 = [2, 3, 2, 4, 6, 3, 3, 6, 4, 3, 2, 4, 2, 3, 3, 4, 3]
        # # coherence_4606 = [2, 3, 2, 4, 12, 3, 3, 12, 4, 3, 2, 4, 2, 3, 3, 4, 3]
        # mean_coherence = 2.788
        # # normalise coherence as a ratio of the mean and add start word
        # plot_histogram(None, [0] + [(i / mean_coherence) ** 2 for i in coherence_4606], label='coherence_dbpedia_4606', xlabels=SAMPLE_4606)
        # # make_boxplot(entity_distribution.values(), label='entity_distribution_boxplot')
