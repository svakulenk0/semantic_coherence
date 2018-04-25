# -*- coding: utf-8 -*-
'''
svakulenko
28 Feb 2018

Iterate over the dialogues from the Ubuntu corpus
'''
import os
import unicodecsv, csv
from collections import Counter
import pickle
import ast
from numpy import array
import random
import json
import numpy as np
from heapq import heappush, heappop

# from trace_relations import trace_relations
from dbpedia_spotlight import annotate_entities
from keras.preprocessing.sequence import pad_sequences
from sample172098 import entity_distribution, word_distribution

PATH = './ubuntu/dialogs'
LATEST_SAMPLE = '291848'
# DIALOGUES_PATH = './ubuntu/annotated_dialogues'
DIALOGUES_PATH = './ubuntu/annotated_dialogues_sample2'  # 172,098
# DIALOGUES_PATH = './ubuntu/annotated_dialogues_only_URIs'
PATH1 = './ubuntu/dialogs/555'
SAMPLE_DIALOG = './ubuntu/dialogs/135/9.tsv'
# VOCAB_ENTITIES_PATH = './ubuntu/vocab_entities.pkl'
VOCAB_ENTITIES_PATH = './%s/entities/vocab.pkl'
VOCAB_WORDS_PATH = './%s/words/vocab.pkl'

dialog_end_symbol = "__dialog_end__"


def create_negative_sample():
    '''
    take 2 random dialogues
    take part of one dialogue and append it to another
    '''
    pass


def translate_dialog_to_lists(dialog_filename):
    """
    from create_ubuntu_dataset.py by Rudolf Kadlec

    Translates the dialog to a list of lists of utterances. In the first
    list each item holds subsequent utterances from the same user. The second level
    list holds the individual utterances.
    :param dialog_filename:
    :return:
    """

    dialog_file = open(dialog_filename, 'r')
    dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t', quoting=csv.QUOTE_NONE)

    # go through the dialog
    first_turn = True
    dialog = []
    same_user_utterances = []
    dialog.append(same_user_utterances)

    for dialog_line in dialog_reader:

        if first_turn:
            last_user = dialog_line[1]
            first_turn = False

        if last_user != dialog_line[1]:
            # user has changed
            same_user_utterances = []
            dialog.append(same_user_utterances)

        same_user_utterances.append(dialog_line[3])

        last_user = dialog_line[1]

    dialog.append([dialog_end_symbol])

    return dialog


def test_translate_dialog_to_lists():
    print translate_dialog_to_lists(PATH + '/10/10007.tsv')


def test_trace_relations():
    dialogue = [' '.join(turn) for turn in translate_dialog_to_lists(SAMPLE_DIALOG)]
    trace_relations(dialogue, True)


def trace_all_dialogues(dir=PATH1):
    # iterate over all the dialogues in the dataset
    for root, dirs, files in os.walk(dir):
        for name in files:
            file_path = os.path.join(root, name)
            print file_path
            dialogue = [' '.join(turn) for turn in translate_dialog_to_lists(file_path)]
            trace_relations(dialogue, True)


def annotate_ubuntu_dialogs(dir=PATH, offset=3):
    '''
    the dialogues are annotated with the lists of the corresponding DBpedia entities of the format:
    'http://dbpedia.org/resource/Sudo'
    '''
    for root, dirs, files in os.walk(dir):
        # iterate over dialogues 
        for name in files[offset:]:
            file_path = os.path.join(root, name)
            annotation_path = os.path.join(DIALOGUES_PATH, '_'.join([root.split('/')[-1], name]))
            print annotation_path
            with open(file_path,"rb") as dialog_file:
                # dialog_reader = unicodecsv.reader(dialog_file)
                dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t', quoting=csv.QUOTE_NONE)
                # annotation_file = unicodecsv.writer(open(annotation_path, 'w'), encoding='utf-8')
                annotation_file = unicodecsv.writer(open(annotation_path, 'w'), encoding='utf-8', delimiter='\t')
                for dialog_line in dialog_reader:
                    # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                    utterance = dialog_line[3]
                    entities = annotate_entities(utterance)
                    # if entities:
                    #     entities = list(set([entity['URI'] for entity in entities]))
                    # dialog_line.append(entities)
                    dialog_line.append(json.dumps(entities))
                    print dialog_line
                    annotation_file.writerow(dialog_line)


def load_dialogues_words(vocabulary, n_dialogues=None, path=DIALOGUES_PATH, vocab_path=VOCAB_WORDS_PATH):
    # generate incorrect examples along the way
    encoded_docs = []
    labels = []
    vocabulary_words = vocabulary.keys()

    dialogues = os.listdir(path)
    if n_dialogues:
        dialogues = dialogues[:n_dialogues]
    
    for file_name in dialogues:
        # extract entities from dialogue and encode them with ids from the vocabulary
        print file_name
        doc_words = []
        encoded_doc = []
        with open(os.path.join(path, file_name),"rb") as dialog_file:
            dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t')
            for dialog_line in dialog_reader:
                # print dialog_line
                # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                annotation_result = dialog_line[4]
                if annotation_result:
                    entities = json.loads(annotation_result)
                    # print entities
                    if entities:
                        for entity in entities:
                            # print entity
                            entity_words = entity['surfaceForm']
                            # skip duplicate entities within the same document
                            for word in entity_words.split():
                                if word not in doc_words:
                                    # encode words with ids
                                    if word in vocabulary_words:
                                        # print len(vocabulary.keys())
                                        encoded_doc.append(vocabulary[word])
                                    else:
                                        encoded_doc.append(vocabulary['<UNK>'])
                                    doc_words.append(word)
        # skip docs with 1 entity
        if len(encoded_doc) > 1:
            print doc_words
            encoded_docs.append(encoded_doc)
            print encoded_doc
            
            labels.append(1)
            # generate counter example by picking as many entities at random from the vocabulary
            # to generate a document of the same # entities as a positive example
            encoded_docs.append(random.sample(xrange(1, len(vocabulary.keys())), len(encoded_doc)))
            labels.append(0)

    # 3 correct + 3 incorrect = 6 docs
    print len(encoded_docs), 'documents encoded'
    padded_docs = pad_sequences(encoded_docs, padding='post')
    # print padded_docs
    return padded_docs, array(labels)


# def sample_negatives_random(sample='sample172098', n_dialogues=None):
#     '''
#     produce 2 datasets (X, y arrays) with word- and entity-based vocabulary encodings
#     '''
    
#     # vocabulary encodings for entities
#     X_path_entities = './%s/entities_X.npy' % sample
#     # vocabulary encodings for words
#     X_path_words = './%s/words_X.npy' % sample
#     y_path = './%s/y.npy' % sample

#     entity_vocabulary = load_vocabulary('./%s/vocab.pkl' % sample)
#     word_vocabulary = load_vocabulary('./%s/vocab_words.pkl' % sample)

#     encoded_docs_entities = []
#     encoded_docs_words = []
#     labels = []

#     dialogues = os.listdir(DIALOGUES_PATH)

#     # limit the number of dialogues to process
#     if n_dialogues:
#         dialogues = dialogues[:n_dialogues]
    
#     for file_name in dialogues:
#         # extract entities from dialogue and encode them with ids from the vocabulary
#         print file_name
#         doc_entities = []
#         doc_words = []
#         encoded_doc_entities = []
#         encoded_doc_words = []
#         with open(os.path.join(DIALOGUES_PATH, file_name),"rb") as dialog_file:
#             dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t')
#             for dialog_line in dialog_reader:
#                 # print dialog_line
#                 # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
#                 entities = json.loads(dialog_line[4])
#                 if entities:
#                     # print entities
#                     for entity in entities:
                        
#                         # encode entity with its URI
#                         entity_URI = entity['URI']
#                         # skip duplicate entities within the same document
#                         if entity_URI not in doc_entities:
#                             # incode entities with ids
#                             if entity_URI in entity_vocabulary:
#                                 # print len(vocabulary.keys())
#                                 encoded_doc_entities.append(entity_vocabulary[entity_URI])
#                             else:
#                                 encoded_doc_entities.append(entity_vocabulary['<UNK>'])
#                             doc_entities.append(entity_URI)
                        
#                         # encode entity with its words
#                         entity_words = entity['surfaceForm']
#                         # skip duplicate entities within the same document
#                         for word in entity_words.split():
#                             if word not in doc_words:
#                                 # encode words with ids
#                                 if word in word_vocabulary:
#                                     # print len(vocabulary.keys())
#                                     encoded_doc_words.append(word_vocabulary[word])
#                                 else:
#                                     encoded_doc_words.append(word_vocabulary['<UNK>'])
#                                 doc_words.append(word)

#         # skip docs with 1 entity
#         if len(encoded_doc_entities) > 1:
#             encoded_docs_entities.append(encoded_doc_entities)
#             encoded_docs_words.append(encoded_doc_words)
#             labels.append(1)
#             # generate incorrect examples along the way by picking as many entities at random from the vocabulary
#             # to generate a document of the same # entities as a positive example
#             encoded_docs_entities.append(random.sample(xrange(1, len(entity_vocabulary.keys())), len(encoded_doc_entities)))
#             encoded_docs_words.append(random.sample(xrange(1, len(word_vocabulary.keys())), len(encoded_doc_words)))
#             labels.append(0)

#     # correct *2
#     print len(encoded_docs_entities), 'documents encoded'
#     X_entities = pad_sequences(encoded_docs_entities, padding='post')
#     X_words = pad_sequences(encoded_docs_words, padding='post')
#     labels = array(labels)

#     print X_entities
#     print X_entities.shape[0], 'dialogues', X_entities.shape[1], 'max entities per dialogue'

#     print X_words
#     print X_words.shape[0], 'dialogues', X_words.shape[1], 'max words per dialogue'

#     print labels

#     # save datasets
#     # save embedding_matrix for entities in the dataset
#     np.save(X_path_entities, X_entities)
#      # save embedding_matrix for words in the dataset
#     np.save(X_path_words, X_words)
#     np.save(y_path, labels)


def encode_turns(dialogue_file_name, entity_vocabulary, word_vocabulary):
    turns = []
    with open(os.path.join(DIALOGUES_PATH, dialogue_file_name),"rb") as dialog_file:
        dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t')
        # collect utterances into turns [0] entity ids [1] word ids
        turn = [[], []]
        author = None
        for dialog_line in dialog_reader:
            # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
            entities = json.loads(dialog_line[4])
            if entities:
                # detect turn switch
                if author and turn and author != dialog_line[1]:
                    turns.append((author, turn))
                    # start tracking new turn
                    turn = [[], []]
                author = dialog_line[1]
                for entity in entities:
                    entity_URI = entity['URI']
                    if entity_URI in entity_vocabulary:
                        turn[0].append(entity_vocabulary[entity_URI])
                    else:
                        turn[0].append(entity_vocabulary['<UNK>'])

                    entity_words = entity['surfaceForm']
                    for word in entity_words.split():
                        # encode words with ids
                        if word in word_vocabulary:
                            # print len(vocabulary.keys())
                            turn[1].append(word_vocabulary[word])
                        else:
                            turn[1].append(word_vocabulary['<UNK>'])
    # for author, utterance in turns:
    #     print author, utterance
    if len(turns) > 1:
        return turns


def pop_random(lst):
    # print lst
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)

# def sample_negatives_horizontal(sample='sample172098', n_dialogues=None):

def add_dialogue_turns(dialog):
    # roll out turns and create word and entity encoded documents
    encoded_doc_entities = []
    for turn in dialog:
        for entity in turn[1][0]:
            if entity not in encoded_doc_entities:
                encoded_doc_entities.append(entity)
    # print encoded_doc_entities
    
    encoded_doc_words = []
    for turn in dialog:
        for entity in turn[1][1]:
            if entity not in encoded_doc_words:
                encoded_doc_words.append(entity)
    # print encoded_doc_words
    
    return encoded_doc_entities, encoded_doc_words


def sample_negatives_random(positive_examples_entities, positive_examples_words):
    '''
    sample negatives from vocabulary distribution
    voc_distr <dict> vocabulary distribution (word/entity: count)
    '''
    entities_dataset = []
    words_dataset = []

    # prepare probabilities from vocabulary counts distribution
    entities = entity_distribution.keys()
    entities_counts = entity_distribution.values()
    entities_probs = [count / float(sum(entities_counts)) for count in entities_counts]

    words = word_distribution.keys()
    words_counts = word_distribution.values()
    words_probs = [count / float(sum(words_counts)) for count in words_counts]

    print '\nGenerating random negatives from the vocabulary distribution'
    
    for i, positive_entities in enumerate(positive_examples_entities):
        n_entities = len(positive_entities)
        # consider only the dialogue with at least 3 detected entities
        if n_entities > 2:
            # sample from entity vocabulary distribution without duplicates
            negative_entities = np.random.choice(entities, replace=False, size=n_entities, p=entities_probs)
            assert len(negative_entities) == n_entities
            entities_dataset.append(positive_entities)
            entities_dataset.append(negative_entities)

            # sample from word vocabulary distribution without duplicates
            positive_words = positive_examples_words[i]
            n_words = len(positive_words)
            negative_words = np.random.choice(words, replace=False, size=n_words, p=words_probs)
            assert len(negative_words) == n_words
            words_dataset.append(positive_words)
            words_dataset.append(negative_words)

    return pad_sequences(entities_dataset, padding='post'), pad_sequences(words_dataset, padding='post')


def encode_positive_examples(sample='sample172098', n_dialogues=None):
    '''
    produce 2 datasets (X, y arrays) with word- and entity-based vocabulary encodings for the positive examples of sequences from the dialogues
    '''
    dialogues = os.listdir(DIALOGUES_PATH)

    # limit the number of dialogues to process
    if n_dialogues:
        dialogues = dialogues[:n_dialogues]

    entity_vocabulary = load_vocabulary(VOCAB_ENTITIES_PATH % sample)
    word_vocabulary = load_vocabulary(VOCAB_WORDS_PATH % sample)

    encoded_docs_entities = []
    encoded_docs_words = []

    for file_name in dialogues:
        # extract entities from dialogue and encode them with ids from the vocabulary
        # print file_name
        encoded_doc_entities = []
        encoded_doc_words = []
        with open(os.path.join(DIALOGUES_PATH, file_name),"rb") as dialog_file:
            dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t')
            for dialog_line in dialog_reader:
                # encode all entities in the dialog
                # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                entities = json.loads(dialog_line[4])
                if entities:
                    # print entities
                    for entity in entities:
                        # encode entity with its URI
                        entity_URI = entity['URI']
                        if entity_URI in entity_vocabulary:
                            # incode entities with ids
                            entity_id = entity_vocabulary[entity_URI]
                            # skip duplicate entities within the same document
                            if entity_id not in encoded_doc_entities:
                                encoded_doc_entities.append(entity_id)

                                # encode entity with its words
                                entity_words = entity['surfaceForm']
                                # skip duplicate entities within the same document
                                for word in entity_words.split():
                                    # encode words with ids
                                    if word in word_vocabulary:
                                        word_id = word_vocabulary[word]
                                        if word_id not in encoded_doc_words:
                                            encoded_doc_words.append(word_id)
            # add encoded dialog to the dataset
            encoded_docs_entities.append(encoded_doc_entities)
            encoded_docs_words.append(encoded_doc_words)

    print len(encoded_docs_entities), 'documents encoded'
    # distribution for the number of entities across dialogues
    # print encoded_docs_entities
    # print encoded_docs_words
    entity_counts = [len(entities) for entities in encoded_docs_entities]
    total_entities = sum(entity_counts)
    print total_entities, 'entities in total'
    # print Counter(entity_counts)
    
    words_counts = [len(words) for words in encoded_docs_words]
    total_words = sum(words_counts)
    print total_words, 'words in total'
    # print Counter(words_counts)
    # frequency distribution of entities in the dataset
    # print Counter([entity for entities in encoded_docs_entities for entity in entities])
    # print Counter([word for words in encoded_docs_words for word in words])
    return encoded_docs_entities, encoded_docs_words


def create_datasets(sample='sample172098', n_dialogues=None):
    '''
    produce 2 datasets (X, y arrays) with word- and entity-based vocabulary encodings
    '''
    entity_vocabulary = load_vocabulary('./%s/vocab.pkl' % sample)
    word_vocabulary = load_vocabulary('./%s/vocab_words.pkl' % sample)

    encoded_docs_entities_vertical = []
    encoded_docs_entities_random = []
    encoded_docs_entities_horizontal = []
    encoded_docs_words_vertical = []
    encoded_docs_words_random = []
    encoded_docs_words_horizontal = []
    labels = []

    dialogues = os.listdir(DIALOGUES_PATH)

    # limit the number of dialogues to process
    if n_dialogues:
        dialogues = dialogues[:n_dialogues]

    turns = []

    for dialogue in dialogues:

    # while len(dialogues) > 1:
    #     # create dialogue pairs by picking non-empty dialogues at random
    #     turns1 = []
    #     while not turns1:
    #         if len(dialogues) < 2:
    #             break
    #         dialogue = pop_random(dialogues)
            dialogue_in_turns = encode_turns(dialogue, entity_vocabulary, word_vocabulary)
            if dialogue_in_turns:
                n_turns = len(dialogue_in_turns)
                if n_turns > 1:
                    heappush(turns, (-n_turns, dialogue_in_turns))

    while turns:
        n_turns1, turns1 = heappop(turns)
        n_turns2, turns2 = heappop(turns)

        # generate 4 dialogues: 2 positive, 2 negative by trancating and mixing utterances
        dialogue1, dialogue2 = [], []
        dialogue12_v, dialogue21_v = [], []
        dialogue12_h, dialogue21_h = [], []

        dialogue_length = min([-n_turns1, -n_turns2])
        
        for i, turn1 in enumerate(turns1[:dialogue_length]):
            # generate positive example: original structure
            dialogue1.append(turn1)
            dialogue2.append(turns2[i])

            # generate vertical negative example
            if i % 2:
                # every 2nd turn mix in utterance from another dialogue
                dialogue21_v.append(turn1)
                dialogue12_v.append(turns2[i])
            else:
                dialogue12_v.append(turn1)
                dialogue21_v.append(turns2[i])

            # generate horizontal negative example by mixing the two dialogues in halves
            if i > dialogue_length / 2:
                # after half of the dialogue mix in utterance from another dialogue
                dialogue21_h.append(turn1)
                dialogue12_h.append(turns2[i])
            else:
                dialogue12_h.append(turn1)
                dialogue21_h.append(turns2[i])

        # print dialogue1, '\n'
        # print dialogue2, '\n'
        # print dialogue12, '\n'
        # print dialogue21, '\n'

        assert len(dialogue1) == len(dialogue2) == len(dialogue12_v) == len(dialogue21_v) == len(dialogue12_h) == len(dialogue21_h)

        # encode the first positive-negative pair
        encoded_doc_entities1, encoded_doc_words1 = add_dialogue_turns(dialogue1)
        # include only dialogues with more than 1 entity
        len_doc1 = len(encoded_doc_entities1)

        if len_doc1 > 1:
            # add positive example
            encoded_docs_entities_vertical.append(encoded_doc_entities1)
            encoded_docs_entities_random.append(encoded_doc_entities1)
            encoded_docs_entities_horizontal.append(encoded_doc_entities1)
            encoded_docs_words_vertical.append(encoded_doc_words1)
            encoded_docs_words_random.append(encoded_doc_words1)
            encoded_docs_words_horizontal.append(encoded_doc_words1)

            # add negative examples
            # vertical
            encoded_doc_entities12_v, encoded_doc_words12_v = add_dialogue_turns(dialogue12_v)
            encoded_docs_entities_vertical.append(encoded_doc_entities12_v[:len_doc1])
            encoded_docs_words_vertical.append(encoded_doc_words12_v[:len(encoded_doc_words1)])
            # generate incorrect examples along the way by picking as many entities at random from the vocabulary distribution TODO
            # to generate a document of the same # entities as a positive example
            encoded_docs_entities_random.append(random.sample(xrange(1, len(entity_vocabulary.keys())), len_doc1))
            encoded_docs_words_random.append(random.sample(xrange(1, len(word_vocabulary.keys())), len(encoded_doc_words1)))
            
            # horizontal
            encoded_doc_entities12_h, encoded_doc_words12_h = add_dialogue_turns(dialogue12_h)
            encoded_docs_entities_horizontal.append(encoded_doc_entities12_h[:len_doc1])
            encoded_docs_words_horizontal.append(encoded_doc_words12_h[:len(encoded_doc_words1)])

            labels.extend([1, 0])

        # encode the second positive-negative pair
        encoded_doc_entities2, encoded_doc_words2 = add_dialogue_turns(dialogue2)
        # include only dialogues with more than 1 entity
        len_doc2 = len(encoded_doc_entities2)
        
        if len_doc2 > 1:
            encoded_docs_entities_vertical.append(encoded_doc_entities2)
            encoded_docs_entities_random.append(encoded_doc_entities2)
            encoded_docs_entities_horizontal.append(encoded_doc_entities2)
            encoded_docs_words_vertical.append(encoded_doc_words2)
            encoded_docs_words_random.append(encoded_doc_words2)
            encoded_docs_words_horizontal.append(encoded_doc_entities2)

            # add negative examples
            encoded_doc_entities21_v, encoded_doc_words21_v = add_dialogue_turns(dialogue21_v)
            encoded_docs_entities_vertical.append(encoded_doc_entities21_v[:len_doc2])
            encoded_docs_words_vertical.append(encoded_doc_words21_v[:len(encoded_doc_words2)])
            # generate incorrect examples along the way by picking as many entities at random from the vocabulary
            # to generate a document of the same # entities as a positive example
            encoded_docs_entities_random.append(random.sample(xrange(1, len(entity_vocabulary.keys())), len_doc2))
            encoded_docs_words_random.append(random.sample(xrange(1, len(word_vocabulary.keys())), len(encoded_doc_words2)))

            # horizontal
            encoded_doc_entities21_h, encoded_doc_words21_h = add_dialogue_turns(dialogue21_h)
            encoded_docs_entities_horizontal.append(encoded_doc_entities21_h[:len_doc1])
            encoded_docs_words_horizontal.append(encoded_doc_words21_h[:len(encoded_doc_words1)])

            labels.extend([1, 0])

        # for dialogue, label in [[dialogue1, 1], [dialogue12, 0], [dialogue2, 1], [dialogue21, 0]]:
        #     encoded_doc_entities, encoded_doc_words = add_dialogue_turns(dialogue)
        #     # include only dialogues with more than 1 entity
        #     if len(encoded_doc_entities) > 1:
        #         encoded_docs_entities.append(encoded_doc_entities)
        #         encoded_docs_words.append(encoded_doc_words)
        #         labels.append(label)

        # print encoded_docs_entities
        # print encoded_docs_words
        # print labels

    assert len(encoded_docs_entities_vertical) == len(encoded_docs_words_vertical) == len(encoded_docs_entities_random) == len(encoded_docs_entities_random) == len(encoded_docs_entities_horizontal) == len(encoded_docs_words_horizontal) == len(labels)
    print len(encoded_docs_entities_vertical), 'documents encoded'
    
    X_entities_vertical = pad_sequences(encoded_docs_entities_vertical, padding='post')
    X_words_vertical = pad_sequences(encoded_docs_words_vertical, padding='post')
    X_entities_horizontal = pad_sequences(encoded_docs_entities_horizontal, padding='post')
    X_words_horizontal = pad_sequences(encoded_docs_words_horizontal, padding='post')
    X_entities_random = pad_sequences(encoded_docs_entities_random, padding='post')
    X_words_random = pad_sequences(encoded_docs_words_random, padding='post')
    labels = array(labels)

    print 'vertical'
    print X_entities_vertical
    print X_entities_vertical.shape[0], 'dialogues', X_entities_vertical.shape[1], 'max entities per dialogue'
    print X_words_vertical
    print X_words_vertical.shape[0], 'dialogues', X_words_vertical.shape[1], 'max words per dialogue'

    print 'horizontal'
    print X_entities_horizontal
    print X_entities_horizontal.shape[0], 'dialogues', X_entities_horizontal.shape[1], 'max entities per dialogue'
    print X_words_horizontal
    print X_words_horizontal.shape[0], 'dialogues', X_words_horizontal.shape[1], 'max words per dialogue'

    print 'random'
    print X_entities_random
    print X_entities_random.shape[0], 'dialogues', X_entities_random.shape[1], 'max entities per dialogue'
    print X_words_random
    print X_words_random.shape[0], 'dialogues', X_words_random.shape[1], 'max words per dialogue'

    print labels

    # save datasets
    np.save('./%s/entities_vertical_X.npy' % sample, X_entities_vertical)
    np.save('./%s/words_vertical_X.npy' % sample, X_words_vertical)
    np.save('./%s/entities_horizontal_X.npy' % sample, X_entities_horizontal)
    np.save('./%s/words_horizontal_X.npy' % sample, X_words_horizontal)
    np.save('./%s/entities_random_X.npy' % sample, X_entities_random)
    np.save('./%s/words_random_X.npy' % sample, X_words_random)
    np.save('./%s/y.npy' % sample, labels)


def load_annotated_dialogues(vocabulary, n_dialogues=None, path=DIALOGUES_PATH, vocab_path=VOCAB_ENTITIES_PATH):
    # generate incorrect examples along the way
    encoded_docs = []
    labels = []
    vocabulary_entities = vocabulary.keys()

    dialogues = os.listdir(path)
    if n_dialogues:
        dialogues = dialogues[:n_dialogues]
    
    for file_name in dialogues:
        # extract entities from dialogue and encode them with ids from the vocabulary
        print file_name
        doc_entities = []
        encoded_doc = []
        with open(os.path.join(path, file_name),"rb") as dialog_file:
            dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t')
            for dialog_line in dialog_reader:
                # print dialog_line
                # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                entities = json.loads(dialog_line[4])
                if entities:
                    # print entities
                    for entity in entities:
                        entitiy_URI = entity['URI']
                        # skip duplicate entities within the same document
                        if entitiy_URI not in doc_entities:
                            # incode entities with ids
                            if entitiy_URI in vocabulary_entities:
                                # print len(vocabulary.keys())
                                encoded_doc.append(vocabulary[entitiy_URI])
                            else:
                                encoded_doc.append(vocabulary['<UNK>'])
                            doc_entities.append(entitiy_URI)

        # skip docs with 1 entity
        if len(encoded_doc) > 1:
            encoded_docs.append(encoded_doc)
            labels.append(1)
            # generate counter example by picking as many entities at random from the vocabulary
            # to generate a document of the same # entities as a positive example
            encoded_docs.append(random.sample(xrange(1, len(vocabulary.keys())), len(encoded_doc)))
            labels.append(0)

    # 3 correct + 3 incorrect = 6 docs
    print len(encoded_docs), 'documents encoded'
    padded_docs = pad_sequences(encoded_docs, padding='post')
    print padded_docs
    return padded_docs, array(labels)


def load_vocabulary(path):
    with open(path, 'rb') as f:
        vocabulary = pickle.load(f)
        print 'Loaded vocabulary with', len(vocabulary.keys()), 'entities'
        return vocabulary


def create_vocabulary_words(sample, n_dialogues=None, path=DIALOGUES_PATH, save_to=VOCAB_WORDS_PATH):
    # entities -> int ids
    vocabulary = {'<UNK>': 0}
    dialogues = os.listdir(path)
    if n_dialogues:
        dialogues = dialogues[:n_dialogues]
    for file_name in dialogues:
        print file_name
        with open(os.path.join(path, file_name),"rb") as dialog_file:
            dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t')
            for dialog_line in dialog_reader:
                # print dialog_line
                # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                try:
                    entities = json.loads(dialog_line[4])
                    # print entities
                    if entities:
                        for entity in entities:
                            # print entity
                            entitiy_words = entity['surfaceForm']
                            for word in entitiy_words.split():
                                if word not in vocabulary.keys():
                                    # print len(vocabulary.keys())
                                    vocabulary[word] = len(vocabulary.keys())
                except:
                    print dialog_line

    # save vocabulary on disk
    with open(save_to%sample, 'wb') as f:
        pickle.dump(vocabulary, f)
    print 'Saved vocabulary with', len(vocabulary.keys()), 'words'


def create_vocabulary(sample, n_dialogues=None, path=DIALOGUES_PATH, save_to=VOCAB_ENTITIES_PATH):
    # entities -> int ids
    vocabulary = {'<UNK>': 0}
    dialogues = os.listdir(path)
    if n_dialogues:
        dialogues = dialogues[:n_dialogues]
    for file_name in dialogues:
        print file_name
        with open(os.path.join(path, file_name),"rb") as dialog_file:
            # dialog_reader = unicodecsv.reader(dialog_file, delimiter=',')
            dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t')
            for dialog_line in dialog_reader:
                try:
                    entities = json.loads(dialog_line[4])
                    # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                    # print entities
                    if entities:
                        for entity in entities:
                            entitiy_URI = entity['URI']
                            if entitiy_URI not in vocabulary:
                                # print len(vocabulary.keys())
                                vocabulary[entitiy_URI] = len(vocabulary.keys())
                except:
                    print dialog_line

    # save vocabulary on disk
    with open(save_to % sample, 'wb') as f:
        pickle.dump(vocabulary, f)
    print 'Saved vocabulary with', len(vocabulary.keys()), 'entities'


def count_ubuntu_dialogs(dir=PATH):
    n_files = 0
    for root, dirs, files in os.walk(dir):
        # iterate over dialogues 
        for name in files:
            file_path = os.path.join(root, name)
            # print file_path
            n_files += 1
    print n_files


def produce_dialog_stats(dir=PATH):
    # iterate over all the dialogues in the dataset
    n_files = 0
    n_annotated_dialogues = 0
    n_turns_dist = []
    n_utterances_dist = []
    n_entities_dist = []
    n_annotated_turns_dist = []
    n_unique_entities_dist = []

    for root, dirs, files in os.walk(dir):

        # iterate over dialogues 
        for name in files:
            file_path = os.path.join(root, name)
            print file_path
            n_files += 1

            # analyse dialogue-file
            n_turns = 0
            n_utterances = 0
            # DBpedia annotation stats
            n_entities = 0
            dialogue_annotated = 0
            n_annotated_turns = 0
            dialogue_entities = []


            # iterate over turns (speakers switch)
            for turn in translate_dialog_to_lists(file_path):
                turn_annotated = 0
                n_turns += 1

                # iterate over utterances in a turn (same speaker)
                for utterance in turn:
                    n_utterances += 1
                    entities = annotate_entities(utterance)
                    if entities:
                        # print entities
                        dialogue_annotated = 1
                        turn_annotated = 1
                        dialogue_entities.extend(entities)
                        n_entities += len(entities)

                n_annotated_turns += turn_annotated

            n_turns_dist.append(n_turns)
            n_utterances_dist.append(n_utterances)

            n_entities_dist.append(n_entities)
            n_annotated_dialogues += dialogue_annotated
            n_annotated_turns_dist.append(n_annotated_turns)

            dialogue_entities = Counter(dialogue_entities)
            # unique entities
            n_unique_entities = len(dialogue_entities.keys())
            n_unique_entities_dist.append(n_unique_entities)

            # print '\n'

    # dataset stats
    print '#Dialogues:', n_files
    # annotation stats
    print '#Annotated dialogues:', n_annotated_dialogues

    print '#Turns per dialogue:', n_turns_dist
    print '#Annotated turns per dialogue:', n_annotated_turns_dist
    print '#Entities per dialogue:', n_entities_dist
    print '#Unique entities per dialogue:', n_unique_entities_dist


def test_load_vocabulary(path):
    vocabulary = load_vocabulary(path)
    # sort by id
    for entity, entity_id in sorted(vocabulary.iteritems(), key=lambda (k,v): (v,k)):
        print  entity, entity_id


if __name__ == '__main__':
    # 1. annotate dialogues with DBpedia entities and save (create dir ./ubuntu/annotated_dialogues)
    # annotate_ubuntu_dialogs()
    
    sample = 'sample172098_new'

    # 2. load all entities and save into a vocabulary dictionary

    # create_vocabulary(sample)
    # test_load_vocabulary(VOCAB_ENTITIES_PATH%sample)

    # create_vocabulary_words(sample)
    # test_load_vocabulary(VOCAB_WORDS_PATH%sample)
    
    # generate dataset
    # encode all positive examples from the dataset
    encoded_docs_entities, encoded_docs_words = encode_positive_examples(sample)
    assert len(encoded_docs_entities) == len(encoded_docs_words)
    # add one random sampled from vocabulary distribution for each positive
    X_entities_random, X_words_random = sample_negatives_random(encoded_docs_entities, encoded_docs_words)
    y = array([1, 0] * (len(X_words_random) / 2))
    assert len(X_entities_random) == len(X_words_random) == len(y)
    print len(X_entities_random)
    print len(X_words_random)
    print y
    np.save('./%s/entities_random_X.npy' % sample, X_entities_random)
    np.save('./%s/words_random_X.npy' % sample, X_words_random)
    np.save('./%s/y.npy' % sample, y)

    # load_annotated_dialogues()
    # load_dialogues_words()
