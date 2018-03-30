# -*- coding: utf-8 -*-
'''
svakulenko
22 Mar 2018

Iterate over the dialogues annotated with entities and extract shortest paths
topk=5, max_length=250000 (unlimited) per new entity
snow-balling
'''
import os
import unicodecsv
import ast

from process_ubuntu_dialogues import DIALOGUES_PATH
from hdt_topk import get_topk_paths

PATH_SHORTEST_PATHS = './ubuntu/paths.txt'

# 18 entities
SAMPLE_4606 = [u'Zip_(file_format)', u'Tar_(computing)', u'Md5sum', u'MD5', u'Ubuntu_(philosophy)', u'MacOS', u'Computer', u'Mac_OS_X_Leopard', u'APT_(Debian)', u'Runlevel', u'Bluetooth', u'Init', u'Gestational_diabetes', u'RC2', u'Salmon_run', u'Strategy_guide', u'Battle', u'Compiz']
SAMPLE_WORDS_4606 = [u'zip', u'file', u'.tar.gz', u'md5sum', u'md5', u'ubuntu', u'OSX', u'computer', u'Leopard', u'apt-get', u'runlevel', u'bluetooth', u'init', u'gdm', u'rc2', u'spawning', u'walkthrough', u'battle', u'compiz']


def annotate_sample(entities=SAMPLE_4606):
    paths = []
    previous_entities = []

    for entity in entities:
        entity = entity.replace(')', '\)').replace('(', '\(').replace(',', '\,')
        print entity
        if entity not in previous_entities:
            if previous_entities:
                # add links to the new entity from all entities mentioned previously through KG relations
                paths.append(get_topk_paths(previous_entities, [entity], k=5, max_length=9))
                print paths
            previous_entities.append(entity)
    print paths


def annotate_shortest_paths(offset=748, source=DIALOGUES_PATH, target=PATH_SHORTEST_PATHS):
    '''
    the dialogues are annotated with shortest paths from DBpedia KG
    '''
    with open(target, 'a') as paths_file:
        # iterate over dialogues
        for file in os.listdir(source)[offset:]:
            previous_entities = []
            paths = []
            with open(os.path.join(source, file),"rb") as dialog_file:
                print file
                dialog_reader = unicodecsv.reader(dialog_file, delimiter=',')
                for dialog_line in dialog_reader:
                    # print dialog_line
                    # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                    entities = dialog_line[4]
                    if entities:
                        # print entities
                        for entity in ast.literal_eval(entities):
                            entity = entity.split('/')[-1].replace(')', '\)').replace('(', '\(').replace(',', '\,')
                            if entity not in previous_entities:
                                # print entity
                                try:
                                    if previous_entities:
                                        # add links to the new entity from all entities mentioned previously through KG relations
                                        paths.extend(get_topk_paths(previous_entities, [entity], k=5, max_length=250000))
                                        print paths
                                    previous_entities.append(entity)
                                except:
                                    pass
            if paths:
                paths_file.write((file + '\t' + '\t'.join(paths)).encode('utf-8') + '\n')


if __name__ == '__main__':
    # annotate_shortest_paths()
    annotate_sample()
