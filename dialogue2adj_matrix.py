# -*- coding: utf-8 -*-
'''
svakulenko
15 Mar 2018

Represent the dialogue concept transitions as an adjacency matrix of a graph.
'''
from process_ubuntu_dialogues import translate_dialog_to_lists
from dbpedia_spotlight import annotate_entities
from hdt_topk import get_topk_paths
from dialogue2dot_graph import escape

SAMPLE_DIALOGUE = './ubuntu/dialogs/135/1.tsv'


def generate_matrix(paths):
    for path in paths:
        print path[0]
        # hops = path[1:-1].split('-<')
        # nhops = len(hops)


def extract_relations(annotated_dialogue):
    paths = []
    previous_entities = []
    for entities in annotated_dialogue:
        if entities:
            for entity in entities:
                entity = escape(entity)
                if entity not in previous_entities:
                    if previous_entities:
                        print previous_entities
                        print entity
                        # find 1 shortest path to the new entity from any of the previous entities 
                        paths.append(get_topk_paths(previous_entities, [entity], k=1))
                    previous_entities.append(entity)
    return paths


def annotate_dialogue(dialogue):
    annotated_dialogue = []
    for n, utterance in enumerate(dialogue):
        annotations = annotate_entities(utterance)
        annotated_dialogue.append(annotations)
    return annotated_dialogue


def test_extract_relations():
    utterances = translate_dialog_to_lists(SAMPLE_DIALOGUE)
    annotated_dialogue = annotate_dialogue(utterances)
    print annotated_dialogue
    print extract_relations(annotated_dialogue)


def test_generate_matrix():
    from sample_paths import sample_paths
    generate_matrix(sample_paths)


if __name__ == '__main__':
    test_extract_relations()
