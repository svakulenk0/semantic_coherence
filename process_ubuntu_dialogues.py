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

# from trace_relations import trace_relations
from dbpedia_spotlight import annotate_entities
from keras.preprocessing.sequence import pad_sequences

PATH = './ubuntu/dialogs'
# PATH_ANNOTATIONS = './ubuntu/annotated_dialogues'
PATH_ANNOTATIONS = './ubuntu/annotated_dialogues_sample'
# PATH_ANNOTATIONS = './ubuntu/annotated_dialogues_only_URIs'
# PATH_ANNOTATIONS = './ubuntu/annotated_dialogues_sample'
PATH1 = './ubuntu/dialogs/555'
SAMPLE_DIALOG = './ubuntu/dialogs/135/9.tsv'
# VOCAB_ENTITIES_PATH = './ubuntu/vocab_entities.pkl'
VOCAB_ENTITIES_PATH = './ubuntu/vocab.pkl'
VOCAB_WORDS_PATH = './ubuntu/vocab_words.pkl'

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
            annotation_path = os.path.join(PATH_ANNOTATIONS, '_'.join([root.split('/')[-1], name]))
            print annotation_path
            with open(file_path,"rb") as dialog_file:
                dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t', quoting=csv.QUOTE_NONE)
                annotation_file = unicodecsv.writer(open(annotation_path, 'w'), encoding='utf-8', delimiter='\t')
                for dialog_line in dialog_reader:
                    # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                    utterance = dialog_line[3]
                    entities = annotate_entities(utterance)
                    # dialog_line.append(entities)
                    dialog_line.append(json.dumps(entities))
                    print dialog_line
                    annotation_file.writerow(dialog_line)


def load_dialogues_words(vocabulary, n_dialogues=None, path=PATH_ANNOTATIONS, vocab_path=VOCAB_WORDS_PATH):
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
        doc_entities = []
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
                            for word in entity_words.split():
                                # skip duplicate entities within the same document
                                if word not in doc_entities:
                                    # encode words with ids
                                    if word in vocabulary_words:
                                        # print len(vocabulary.keys())
                                        encoded_doc.append(vocabulary[word])
                                    else:
                                        encoded_doc.append(vocabulary['<UNK>'])
                            doc_entities.append(entity_words)
        # skip docs with 1 entity
        if len(encoded_doc) > 1:
            print doc_entities
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


def load_annotated_dialogues(vocabulary, n_dialogues=None, path=PATH_ANNOTATIONS, vocab_path=VOCAB_ENTITIES_PATH):
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
            dialog_reader = unicodecsv.reader(dialog_file, delimiter=',')
            for dialog_line in dialog_reader:
                # print dialog_line
                # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                entities = dialog_line[4]
                if entities:
                    # print entities
                    for entity in ast.literal_eval(entities):
                        # skip duplicate entities within the same document
                        if entity not in doc_entities:
                            # incode entities with ids
                            if entity in vocabulary_entities:
                                # print len(vocabulary.keys())
                                encoded_doc.append(vocabulary[entity])
                            else:
                                encoded_doc.append(vocabulary['<UNK>'])
                            doc_entities.append(entity)

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


def create_vocabulary_words(n_dialogues=None, path=PATH_ANNOTATIONS, save_to=VOCAB_WORDS_PATH):
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


def create_vocabulary(n_dialogues=None, path=PATH_ANNOTATIONS, save_to=VOCAB_ENTITIES_PATH):
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
                print dialog_line
                # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                entities = dialog_line[4]
                # print entities
                if entities:
                    for entity in ast.literal_eval(entities):
                        if entity not in vocabulary.keys():
                            # print len(vocabulary.keys())
                            vocabulary[entity] = len(vocabulary.keys())

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
        # map words to ids for word vectors
    create_vocabulary_words()
    test_load_vocabulary(VOCAB_WORDS_PATH)
    # 3. load annotated dialogues convert entities to ids using vocabulary
    # load_annotated_dialogues()
