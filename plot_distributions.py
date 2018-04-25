# -*- coding: utf-8 -*-
'''
svakulenko
4 Apr 2018

Generate plots of distributions from dictionary or Counter objects
'''
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# from sample291848 import word2vec_distribution, entity_distribution, get_top
from prepare_dataset import VOCAB_WORDS_PATH, VOCAB_ENTITIES_PATH
from parse_paths import get_shortest_path_distribution
# from annotate_shortest_paths import SAMPLE_4606

mentioned_entities = [(u'Ubuntu_(philosophy)', 1605), (u'Sudo', 708), (u'Booting', 676), (u'APT_(Debian)', 405), (u'Live_CD', 314), (u'Disk_partitioning', 302), (u'Linux_kernel', 293), (u'Computer', 285), (u'Web_server', 254), (u'Gnome', 252), (u'Laptop', 252), (u'Microsoft_Windows', 243), (u'File_system', 235), (u'Dev_Alahan', 234), (u'Hard_disk_drive', 206), (u'Linux', 191), (u'Graphical_user_interface', 187), (u'Superuser', 165), (u'Pastebin', 160), (u'Software', 159)]
ext_entities = [(u'Ubuntu_(operating_system)', 1058), (u'Linux', 725), (u'Microsoft_Windows', 208), (u'FreeBSD', 175), (u'Smartphone', 171), (u'Canonical_(company)', 163), (u'Coronation_Street', 162), (u'Superuser', 147), (u'Debian', 135), (u'Manchester', 123), (u'London', 123), (u'Wi-Fi', 118), (u'Software', 112), (u'Command-line_interface', 110), (u'Computer_data_storage', 110), (u'GNU_General_Public_License', 107), (u'Zulu_language', 104), (u'Android_(operating_system)', 101), (u'United_States', 90), (u'NetBSD', 89)]
ext_relations = [(u'ontology/wikiPageWikiLink', 51014), (u'gold/hypernym', 319), (u'ontology/genre', 178), (u'ontology/operatingSystem', 140), (u'rdf-schema#seeAlso', 116), (u'ontology/license', 97), (u'ontology/product', 87), (u'ontology/programmingLanguage', 79), (u'ontology/wikiPageDisambiguates', 78), (u'ontology/wikiPageRedirects', 50), (u'ontology/developer', 43), (u'ontology/industry', 43), (u'ontology/computingPlatform', 29), (u'ontology/language', 28), (u'ontology/location', 26), (u'owl#differentFrom', 18), (u'ontology/author', 18), (u'ontology/foundationPlace', 15), (u'ontology/type', 14), (u'ontology/influencedBy', 12)]

# cosine distances 0.0-1.0
# word embeddings: GloVE + word2vec
GloVe_vertical = [11509, 6652, 9555, 27697, 54182, 77746, 88651, 74241, 48508, 17630]
GloVe_positive = [19000, 8124, 13197, 34648, 63014, 83102, 89145, 70909, 45038, 16073]
GloVe_random = [1, 18, 98, 367, 1303, 5520, 26119, 95426, 155269, 98509]
GloVe_distribution = [59, 2593, 6665, 17393, 42599, 74088, 97714, 97402, 69576, 29248]
GloVe_disorder = [19000, 8115, 13108, 33708, 61088, 78396, 86750, 71407, 48226, 20261]
GloVe_horizontal = [10210, 6493, 10886, 28138, 55762, 81742, 94374, 79271, 51700, 17998]

GloVe_distrs = [GloVe_positive, GloVe_disorder, GloVe_distribution, GloVe_vertical, GloVe_horizontal, GloVe_random]

word2vec_vertical = [11120, 163, 5355, 19780, 37550, 48991, 65377, 81035, 75907, 24791] # 451960 pairs/comparisons
word2vec_positive = [18349, 327, 7103, 24255, 42979, 51687, 68202, 82019, 74378, 23773]
word2vec_random = [0, 5, 50, 466, 3158, 11899, 30569, 70326, 125947, 69656]
word2vec_distribution = [14, 81, 3333, 15986, 39361, 50900, 70742, 86928, 88356, 35213]
word2vec_disorder = [18348, 328, 7046, 24104, 42688, 51694, 67632, 79831, 72891, 27686]
word2vec_horizontal = [9772, 215, 5452, 20515, 40804, 50883, 68756, 85603, 80029, 25785]

word2vec_distrs = [word2vec_positive, word2vec_disorder, word2vec_distribution, word2vec_vertical, word2vec_horizontal, word2vec_random]

# entity embeddings: Rdf2Vec
kg_vertical = [384, 3990, 20635, 39383, 62200, 77296, 73710, 30190, 3394, 79]
kg_positive = [625, 5781, 26534, 45153, 65757, 78756, 74902, 30036, 3386, 67]
kg_random = [49, 271, 2984, 14452, 38735, 76342, 106854, 54796, 7222, 232]
kg_distribution = [127, 1930, 17201, 38645, 68363, 89761, 78030, 31202, 3688, 89]
kg_disorder = [625, 5790, 26404, 44793, 64898, 81216, 74175, 29429, 3580, 87]
kg_horizontal = [421, 4108, 21346, 41436, 65803, 81052, 76780, 30997, 3412, 66]

kg_distrs = [kg_positive, kg_disorder, kg_distribution, kg_vertical, kg_horizontal, kg_random]

colors = ['b', 'g', 'r', 'y', 'c', 'm']


def print_counter(items):
    for entity, count in items:
        print entity, count


def make_boxplot(data, label='new'):
    plt.boxplot(data)
    plt.show()


def plot_histograms(lists, label, w=0.015, cut=6):
    ax = plt.subplot(111)
    x = np.arange(0.0, 1.0, 0.1)  # [:cut]
    xs = [x-2*w, x-w, x, x+w, x+2*w, x+3*w]
    for i, _list in enumerate(lists):
        # normalize vectors
        norm_list = [ count/float(sum(_list)) for count in _list ]
        ax.bar(xs[i], norm_list, width=w, color=colors[i], align='center')
    ax.autoscale(tight=True)
    plt.xticks(x)

    # colors legend
    NA = mpatches.Patch(color='blue', label='True positive')
    EU = mpatches.Patch(color='green', label='Sequence disorder')
    AP = mpatches.Patch(color='red', label='Vocabulary distribution')
    SA = mpatches.Patch(color='y', label='Vertical split')
    SA2 = mpatches.Patch(color='c', label='Horizontal split')
    SA3 = mpatches.Patch(color='m', label='Random uniform')
    plt.legend(handles=[NA, EU, AP, SA, SA2, SA3], loc=2)

    plt.show()
    # plt.savefig('distrs_%s.pdf'%label)


def plot_shortest_path_distribution(label='shortest_paths', w=0.15):
    lists = get_shortest_path_distribution()
    print lists
    ax = plt.subplot(111)
    x_labels = [1, 2, 3, 4, 5, float('inf')]
    x = np.arange(1, 7, 1)
    xs = [x-2*w, x-w, x, x+w, x+2*w, x+3*w]
    for i, _list in enumerate(lists):
        ax.bar(xs[i], _list, width=w, color=colors[i], align='center')
    ax.autoscale(tight=True)
    plt.xticks(x, x_labels)

    # colors legend
    NA = mpatches.Patch(color='blue', label='True positive')
    EU = mpatches.Patch(color='green', label='Sequence disorder')
    AP = mpatches.Patch(color='red', label='Vocabulary distribution')
    SA = mpatches.Patch(color='y', label='Vertical split')
    SA2 = mpatches.Patch(color='c', label='Horizontal split')
    SA3 = mpatches.Patch(color='m', label='Random uniform')
    # plt.legend(handles=[NA, EU, AP, SA, SA2, SA3], loc=2)

    plt.show()
    # plt.savefig('distrs_%s.pdf'%label)


def plot_histogram(x, y, size=(30, 5), label='new', xlabels=None, flip=False):
    '''
    from https://stackoverflow.com/questions/21195179/plot-a-histogram-from-a-dictionary
    '''
    print y
    plt.figure(figsize=size)
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
            plt.xticks(x, xlabels, rotation='vertical')  # , rotation='vertical'
    plt.yscale('log', nonposy='clip')
    plt.tight_layout()
    plt.show()
    # plt.savefig('distr_%s.pdf'%label)


def plot_cosine_subplot(lists, ax, w=0.015):
    x = np.arange(0.0, 1.0, 0.1)
    xs = [x - 2 * w, x - w, x, x + w, x + 2 * w, x + 3 * w]
    for i, _list in enumerate(lists):
        # normalize vectors
        norm_list = [ count / float(sum(_list)) for count in _list ]
        ax.bar(xs[i], norm_list, width=w, color=colors[i], align='center')
    ax.autoscale(tight=True)
    # plt.xticks(x)


def plot_path_subplot(ax, w=0.15):
    lists = get_shortest_path_distribution()
    # print lists
    x_labels = [1, 2, 3, 4, 5, float('inf')]
    x = np.arange(1, 7, 1)
    xs = [x - 2 * w, x - w, x, x + w, x + 2 * w, x + 3 * w]
    for i, _list in enumerate(lists):
        ax.bar(xs[i], _list, width=w, color=colors[i], align='center')
    ax.autoscale(tight=True)
    plt.xticks(x, x_labels)


def plot_subplots():
    # 4 distribution plots for the paper fig.
    # fig = plt.figure(figsize=(15, 15), sharey=True)
    fig, ax = plt.subplots(2, 2, figsize=(15, 15), sharey=True)
    axes = ax.flatten()
    # word2vec
    # ax = plt.subplot(2, 2, 1)
    plot_cosine_subplot(word2vec_distrs, axes[0])
    axes[0].set_title("Word2Vec embeddings")
    
    # colors legend
    NA = mpatches.Patch(color='blue', label='True positive')
    EU = mpatches.Patch(color='green', label='Sequence disorder')
    AP = mpatches.Patch(color='red', label='Vocabulary distribution')
    SA = mpatches.Patch(color='y', label='Vertical split')
    SA2 = mpatches.Patch(color='c', label='Horizontal split')
    SA3 = mpatches.Patch(color='m', label='Random uniform')
    axes[0].legend(handles=[NA, EU, AP, SA, SA2, SA3], loc=2)
    
    # RDF2Vec
    # ax = plt.subplot(2, 2, 2)
    plot_cosine_subplot(kg_distrs, axes[1])
    axes[1].set_title("RDF2Vec embeddings")
    # GloVe
    # ax = plt.subplot(2, 2, 3)
    plot_cosine_subplot(GloVe_distrs, axes[2])
    axes[2].set_title("GloVe embeddings")
    # axes[2].legend(handles=[NA, EU, AP, SA, SA2, SA3], loc=2)

    # KG shortest path
    # ax = plt.subplot(2, 2, 4)
    plot_path_subplot(axes[3])
    axes[3].set_title("DBpedia+Wikidata shortest paths")
    # spacing vertical
    plt.subplots_adjust(hspace=0.15)

    # plt.show()
    plt.savefig('all_distrs.pdf')

if __name__ == '__main__':
    # plot_histograms(kg_distrs, 'kg_distances')
    # plot_histograms(word2vec_distrs, 'word2vec_distances')
    # plot_histograms(GloVe_distrs, 'word2vec_distances')
    # plot_shortest_path_distribution()
    
    plot_subplots()

    # print_counter(mentioned_entities)
    # print_counter(ext_entities)
    # plot_histogram(n_entities_per_dialogue.keys(), n_entities_per_dialogue.values(), label='n_entities_per_dialogue')
    # n = 5
    
    # items = ext_relations
    # label='ext_relations'
    # labels = [entity for entity, count in items]
    # counts = [count for entity, count in items]
    # plot_histogram(None, counts, (10, 5), label, xlabels=labels)
    
    # plot_histogram(None, entity_distribution.most_common()[:n], (15, 30), label='entity_distribution', xlabels=get_top(entity_distribution, VOCAB_ENTITIES_PATH, n))
        # coherence_4606 = [2, 3, 2, 4, 6, 3, 3, 6, 4, 3, 2, 4, 2, 3, 3, 4, 3]
        # # coherence_4606 = [2, 3, 2, 4, 12, 3, 3, 12, 4, 3, 2, 4, 2, 3, 3, 4, 3]
        # mean_coherence = 2.788
        # # normalise coherence as a ratio of the mean and add start word
        # plot_histogram(None, [0] + [(i / mean_coherence) ** 2 for i in coherence_4606], label='coherence_dbpedia_4606', xlabels=SAMPLE_4606)
        # # make_boxplot(entity_distribution.values(), label='entity_distribution_boxplot')
