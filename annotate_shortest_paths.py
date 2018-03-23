# -*- coding: utf-8 -*-
'''
svakulenko
22 Mar 2018

Iterate over the dialogues annotated with entities and extract shortest paths
'''
import os
import unicodecsv
import ast

from process_ubuntu_dialogues import PATH_ENTITIES
from hdt_topk import get_topk_paths

PATH_SHORTEST_PATHS = './ubuntu/paths.txt'


def annotate_shortest_paths(source=PATH_ENTITIES, target=PATH_SHORTEST_PATHS):
    '''
    the dialogues are annotated with shortest paths from DBpedia KG
    '''
    with open(target, 'w') as paths_file:
        # iterate over dialogues
        for file in os.listdir(source):
            previous_entities = []
            paths = []
            with open(os.path.join(source, file),"rb") as dialog_file:
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
    annotate_shortest_paths()
