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

PATH = './ubuntu/dialogs'
# DIALOGUES_PATH = './ubuntu/annotated_dialogues'
DIALOGUES_PATH = './ubuntu/annotated_dialogues_sample2'  # 172,098
# DIALOGUES_PATH = './ubuntu/annotated_dialogues_only_URIs'
PATH1 = './ubuntu/dialogs/555'
SAMPLE_DIALOG = './ubuntu/dialogs/135/9.tsv'
# VOCAB_ENTITIES_PATH = './ubuntu/vocab_entities.pkl'
VOCAB_ENTITIES_PATH = './sample172098/vocab.pkl'
VOCAB_WORDS_PATH = './sample172098/vocab_words.pkl'

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


def sample_negatives_random(sample='sample172098', n_dialogues=None):
    '''
    produce 2 datasets (X, y arrays) with word- and entity-based vocabulary encodings
    '''
    
    # vocabulary encodings for entities
    X_path_entities = './%s/entities_X.npy' % sample
    # vocabulary encodings for words
    X_path_words = './%s/words_X.npy' % sample
    y_path = './%s/y.npy' % sample

    entity_vocabulary = load_vocabulary('./%s/vocab.pkl' % sample)
    word_vocabulary = load_vocabulary('./%s/vocab_words.pkl' % sample)

    encoded_docs_entities = []
    encoded_docs_words = []
    labels = []

    dialogues = os.listdir(DIALOGUES_PATH)

    # limit the number of dialogues to process
    if n_dialogues:
        dialogues = dialogues[:n_dialogues]
    
    for file_name in dialogues:
        # extract entities from dialogue and encode them with ids from the vocabulary
        print file_name
        doc_entities = []
        doc_words = []
        encoded_doc_entities = []
        encoded_doc_words = []
        with open(os.path.join(DIALOGUES_PATH, file_name),"rb") as dialog_file:
            dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t')
            for dialog_line in dialog_reader:
                # print dialog_line
                # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                entities = json.loads(dialog_line[4])
                if entities:
                    # print entities
                    for entity in entities:
                        
                        # encode entity with its URI
                        entity_URI = entity['URI']
                        # skip duplicate entities within the same document
                        if entity_URI not in doc_entities:
                            # incode entities with ids
                            if entity_URI in entity_vocabulary:
                                # print len(vocabulary.keys())
                                encoded_doc_entities.append(entity_vocabulary[entity_URI])
                            else:
                                encoded_doc_entities.append(entity_vocabulary['<UNK>'])
                            doc_entities.append(entity_URI)
                        
                        # encode entity with its words
                        entity_words = entity['surfaceForm']
                        # skip duplicate entities within the same document
                        for word in entity_words.split():
                            if word not in doc_words:
                                # encode words with ids
                                if word in word_vocabulary:
                                    # print len(vocabulary.keys())
                                    encoded_doc_words.append(word_vocabulary[word])
                                else:
                                    encoded_doc_words.append(word_vocabulary['<UNK>'])
                                doc_words.append(word)

        # skip docs with 1 entity
        if len(encoded_doc_entities) > 1:
            encoded_docs_entities.append(encoded_doc_entities)
            encoded_docs_words.append(encoded_doc_words)
            labels.append(1)
            # generate incorrect examples along the way by picking as many entities at random from the vocabulary
            # to generate a document of the same # entities as a positive example
            encoded_docs_entities.append(random.sample(xrange(1, len(entity_vocabulary.keys())), len(encoded_doc_entities)))
            encoded_docs_words.append(random.sample(xrange(1, len(word_vocabulary.keys())), len(encoded_doc_words)))
            labels.append(0)

    # correct *2
    print len(encoded_docs_entities), 'documents encoded'
    X_entities = pad_sequences(encoded_docs_entities, padding='post')
    X_words = pad_sequences(encoded_docs_words, padding='post')
    labels = array(labels)

    print X_entities
    print X_entities.shape[0], 'dialogues', X_entities.shape[1], 'max entities per dialogue'

    print X_words
    print X_words.shape[0], 'dialogues', X_words.shape[1], 'max words per dialogue'

    print labels

    # save datasets
    # save embedding_matrix for entities in the dataset
    np.save(X_path_entities, X_entities)
     # save embedding_matrix for words in the dataset
    np.save(X_path_words, X_words)
    np.save(y_path, labels)


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


def sample_negatives_vertical(sample='sample172098', n_dialogues=None):
    '''
    produce 2 datasets (X, y arrays) with word- and entity-based vocabulary encodings
    '''
    
    # vocabulary encodings for entities
    X_path_entities = './%s/entities_X_vertical.npy' % sample
    # vocabulary encodings for words
    X_path_words = './%s/words_X_vertical.npy' % sample
    y_path = './%s/y_vertical.npy' % sample

    entity_vocabulary = load_vocabulary('./%s/vocab.pkl' % sample)
    word_vocabulary = load_vocabulary('./%s/vocab_words.pkl' % sample)

    encoded_docs_entities = []
    encoded_docs_words = []
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
        dialogue1, dialogue2, dialogue12, dialogue21 = [], [], [], []
        dialogue_length = min([-n_turns1, -n_turns2])
        
        for i, turn1 in enumerate(turns1[:dialogue_length]):
            # generate positive example: original structure
            dialogue1.append(turn1)
            dialogue2.append(turns2[i])

            # generate negative example
            if i % 2:
                # every 2nd turn mix in utterance from another dialogue
                dialogue21.append(turn1)
                dialogue12.append(turns2[i])
            else:
                dialogue12.append(turn1)
                dialogue21.append(turns2[i])

        # print dialogue1, '\n'
        # print dialogue2, '\n'
        # print dialogue12, '\n'
        # print dialogue21, '\n'

        assert len(dialogue1) == len(dialogue2) == len(dialogue12) == len(dialogue21)

        # encode the first positive-negative pair
        encoded_doc_entities, encoded_doc_words = add_dialogue_turns(dialogue1)
        # include only dialogues with more than 1 entity
        len_doc = len(encoded_doc_entities)
        if len_doc > 1:
            encoded_docs_entities.append(encoded_doc_entities)
            encoded_docs_words.append(encoded_doc_words)
            # add negative example
            encoded_doc_entities12, encoded_doc_words12 = add_dialogue_turns(dialogue12)
            encoded_docs_entities.append(encoded_doc_entities12[:len_doc])
            encoded_docs_words.append(encoded_doc_words12[:len(encoded_doc_words)])
            labels.extend([1, 0])

        # encode the second positive-negative pair
        encoded_doc_entities, encoded_doc_words = add_dialogue_turns(dialogue2)
        # include only dialogues with more than 1 entity
        len_doc = len(encoded_doc_entities)
        if len_doc > 1:
            encoded_docs_entities.append(encoded_doc_entities)
            encoded_docs_words.append(encoded_doc_words)
            # add negative example
            encoded_doc_entities21, encoded_doc_words21 = add_dialogue_turns(dialogue21)
            encoded_docs_entities.append(encoded_doc_entities21[:len_doc])
            encoded_docs_words.append(encoded_doc_words21[:len(encoded_doc_words)])
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

    assert len(encoded_docs_entities) == len(encoded_docs_words) == len(labels)
    print len(encoded_docs_entities), 'documents encoded'
    
    X_entities = pad_sequences(encoded_docs_entities, padding='post')
    X_words = pad_sequences(encoded_docs_words, padding='post')
    labels = array(labels)

    print X_entities
    print X_entities.shape[0], 'dialogues', X_entities.shape[1], 'max entities per dialogue'

    print X_words
    print X_words.shape[0], 'dialogues', X_words.shape[1], 'max words per dialogue'

    print labels

    # save datasets
    # save embedding_matrix for entities in the dataset
    # np.save(X_path_entities, X_entities)
    # # save embedding_matrix for words in the dataset
    # np.save(X_path_words, X_words)
    # np.save(y_path, labels)


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


def load_vocabulary(path=VOCAB_ENTITIES_PATH):
    with open(path, 'rb') as f:
        vocabulary = pickle.load(f)
        print 'Loaded vocabulary with', len(vocabulary.keys()), 'entities'
        return vocabulary


def create_vocabulary_words(n_dialogues=None, path=DIALOGUES_PATH, save_to=VOCAB_WORDS_PATH):
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
    with open(save_to, 'wb') as f:
        pickle.dump(vocabulary, f)
    print 'Saved vocabulary with', len(vocabulary.keys()), 'words'


def create_vocabulary(n_dialogues=None, path=DIALOGUES_PATH, save_to=VOCAB_ENTITIES_PATH):
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
    with open(save_to, 'wb') as f:
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
    # 2. load all entities and save into a vocabulary dictionary
    # create_vocabulary()
    # test_load_vocabulary(VOCAB_ENTITIES_PATH)

    # create_vocabulary_words()
    # test_load_vocabulary(VOCAB_WORDS_PATH)
    
    # generate datasets 
    sample = 'sample172098'
    # sample_negatives_random(sample)
    # sample_negatives_horizontal(sample)
    sample_negatives_vertical(sample)

    # load_annotated_dialogues()
    # load_dialogues_words()
