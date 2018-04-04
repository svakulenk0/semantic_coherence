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

ext_entities = [(u'Ubuntu', 2220), (u'Linux', 937), (u'Sudo', 637), (u'Booting', 605), (u'Microsoft_Windows', 434), (u'Computer', 325), (u'Superuser', 311), (u'Live_CD', 306), (u'Disk_partitioning', 282), (u'Linux_kernel', 266), (u'Software', 255), (u'File_system', 252), (u'Laptop', 249), (u'Graphical_user_interface', 249), (u'Debian', 241), (u'Web_server', 227), (u'Hard_disk_drive', 218), (u'Gnome', 216), (u'Dev_Alahan', 199), (u'Command-line_interface', 199)]
ext_relations = [(u'ontology/wikiPageWikiLink', 51014), (u'gold/hypernym', 319), (u'ontology/genre', 178), (u'ontology/operatingSystem', 140), (u'rdf-schema#seeAlso', 116), (u'ontology/license', 97), (u'ontology/product', 87), (u'ontology/programmingLanguage', 79), (u'ontology/wikiPageDisambiguates', 78), (u'ontology/wikiPageRedirects', 50), (u'ontology/developer', 43), (u'ontology/industry', 43), (u'ontology/computingPlatform', 29), (u'ontology/language', 28), (u'ontology/location', 26), (u'owl#differentFrom', 18), (u'ontology/author', 18), (u'ontology/foundationPlace', 15), (u'ontology/type', 14), (u'ontology/influencedBy', 12)]


def print_counter(items):
    for entity, count in items:
        print entity, count


def make_boxplot(data, label='new'):
    plt.boxplot(data)
    plt.show()


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


if __name__ == '__main__':
    print_counter(ext_relations)
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
