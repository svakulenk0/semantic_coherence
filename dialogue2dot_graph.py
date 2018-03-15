# -*- coding: utf-8 -*-
'''
svakulenko
15 Mar 2018

Represent the dialogue concept transitions as a graph in DOT language.
The format is as follows:

graph graphname {
     // This attribute applies to the graph itself
     size="1,1";
     // The label attribute can be used to change the label of a node
     a [label="Foo"];
     // Here, the node shape is changed.
     b [shape=box];
     // These edges both have different line properties
     a -- b -- c [color=blue];
     b -- d [style=dotted];
     // [style=invis] hides a node.
   }

Source: https://en.wikipedia.org/wiki/DOT_(graph_description_language)

To plot the graph with GraphViz run:
     dot -Tpng DocName.dot -o DocName.png
'''
from analyse_ubuntu_dialogues import translate_dialog_to_lists
from dbpedia_spotlight import annotate_entities
from hdt_topk import get_topk_paths

SAMPLE_DIALOG = '../ubuntu-ranking-dataset-creator/src/dialogs/135/9.tsv'
DOT_GRAPH_TEMPLATE = '''
digraph dialogue {
     size="1,1";
     %s
   }
'''
NODE_LABEL = 'A%d [label="%s"];'
EDGE_LABEL = 'A%s -> A%s [label="%d"];'


def dialogue2dot_graph(dialogue, entities={}, entity_ids={}):
    previous_entity_id = None
    entity_count = 1  # encode the entity with a unique id in the entity dictionary
    for n, utterance in enumerate(dialogue):
        annotations = annotate_entities(utterance)
        if annotations:
             for entity in annotations:
                 # add new entity to the dictionary
                 if entity not in entity_ids.keys():
                     entities[entity_count] = entity
                     entity_ids[entity] = entity_count
                     # create node in the graph
                     print NODE_LABEL % (entity_count, entity)
                     entity_count += 1
                 
                 entity_id = entity_ids[entity]
                 if previous_entity_id:
                      # create edge from the previous entity
                      print EDGE_LABEL % (previous_entity_id, entity_id, n+1)
                 previous_entity_id = entity_id


def test_dialogue2dot_graph():
    dialogue2dot_graph(translate_dialog_to_lists(SAMPLE_DIALOG))


if __name__ == '__main__':
     test_dialogue2dot_graph()
