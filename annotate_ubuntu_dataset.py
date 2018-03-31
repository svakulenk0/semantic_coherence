# -*- coding: utf-8 -*-
'''
svakulenko
31 Mar 2018

Process Dbpedia entity annotations of Ubuntu dialogues dataset
'''
import os
import unicodecsv
import json
from collections import Counter

DIALOGUES_PATH = './ubuntu/annotated_dialogues'
ANNOTATION_FILE = 'ubuntu_dialogues_spotlight_annotation.jl'


def collect_entity_annotations(path=DIALOGUES_PATH, n_dialogues=None):
    '''
    produce JSON lines for each dialogue from the annotatated sample
    jsonlines.org
    '''

    dialogues = os.listdir(path)
    # limit number of dialogues to process to a smaller sample
    if n_dialogues:
        dialogues = dialogues[:n_dialogues]

    with open(ANNOTATION_FILE, 'w') as outfile:
        # iterate over dialogue-files
        for file_name in dialogues:
            # processing a single dialogue
            # create annotation as a dictionary
            annotation = {'file_name': file_name, 'turns': [], 'utterance_ids': [],
                          'entity_URIs': [], 'surface_forms': [], 'supports': [], 'similarity_scores': [] }
            # encoding nicknames of the dialogue participants as int
            participants = {}

            with open(os.path.join(path, file_name),"rb") as dialog_file:
                dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t')
                # collect all entities and their attributes from utterances lines
                for i, dialog_line in enumerate(dialog_reader):
                    # utterance
                    # dialog line: [0] timestamp [1] sender [2] recepeint [3] utterance [4] entities
                    # print dialog_line
                    try:
                        entities = json.loads(dialog_line[4])
                        # store utterance annotation only if there are any entities regognised in it
                        if entities:
                            for entity in entities:
                                # store only new entities that were not mentioned in the conversation previously
                                entity_URI = entity['URI']
                                if entity_URI not in annotation['entity_URIs']:
                                    annotation['entity_URIs'].append(entity_URI)
                                    # 1) utterance attributes
                                    annotation['utterance_ids'].append(i)
                                    sender = dialog_line[1]
                                    if sender not in participants:
                                        participants[sender] = len(participants)
                                    annotation['turns'].append(participants[sender])
                                    # 2) entity attributes: collect entity attribute in the annotation dictionary
                                    annotation['surface_forms'].append(entity['surfaceForm'])
                                    annotation['supports'].append(entity['support'])
                                    annotation['similarity_scores'].append(entity['similarityScore'])
                    except:
                        print dialog_line
            # distribution of annotated entities between the participants
            entity_distribution = Counter(annotation['turns']).values()
            # save annotation only if we have new entities contribution from both participants
            if 0 not in entity_distribution:
                annotation['participants_entities'] = entity_distribution
                # produce annotation summary
                # number of annotated entities in total
                annotation['n_entities'] = len(annotation['entity_URIs'])
                # number of annotated utterances
                annotation['n_utterances'] = len(set(annotation['utterance_ids']))
                # number of participants whose turns are annotated
                annotation['n_participants'] = len(set(annotation['turns']))
                
                # write annotation as a json line
                json.dump(annotation, outfile)
                outfile.write("\n")

            # log progress
            print entity_distribution

if __name__ == '__main__':
    collect_entity_annotations()
