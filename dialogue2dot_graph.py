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
     dot -Tpng sample_dialogue.dot -o sample_dialogue.png
     dot -Tpng sample_dialogue_ext.dot -o sample_dialogue_ext.png
'''
import re

from analyse_ubuntu_dialogues import translate_dialog_to_lists
from dbpedia_spotlight import annotate_entities
from hdt_topk import get_topk_paths

SAMPLE_DIALOGUE = '../ubuntu-ranking-dataset-creator/src/dialogs/135/9.tsv'

DOT_GRAPH_TEMPLATE = '''
digraph dialogue {
     size="1,1";
     %s
   }
'''
NODE_LABEL = 'A%d [label="%s"];'
EDGE_LABEL = 'A%s -> A%s [label="%s"];'
EXTERNAL_NODE = 'A%d [label="%s" fontcolor=red];'

ESCAPE_CHARS = {"'":  r"\'", "(":  r"\(", ")":  r"\)"}

# PATH_PATTERN = r'''^\[(.+)-<(.+)>-(.+)\]$'''
# PATH_PATTERN = re.compile(r'''^\[(.+)-<(.+)>-(.+)(?R)?\]$''')

def escape(string):
     '''
     Escape special charecters: ', (, )
     '''
     new_string = ''
     for char in string:
          if char in ESCAPE_CHARS.keys():
             new_string += ESCAPE_CHARS[char]
          else:
             new_string += char
     return new_string


def dialogue2dot_graph(dialogue, entities={}, entity_ids={}):
    # entity chain
    previous_entities = []
    entity_count = 1  # encode the entity with a unique id in the entity dictionary
    annotated_utterance_id = 0
    for n, utterance in enumerate(dialogue):
        annotations = annotate_entities(utterance)
        if annotations:
            annotated_utterance_id += 1
            for entity in annotations:
               entity = escape(entity)
                 # add new entity to the dictionary
               if entity not in entity_ids.keys():
                     entities[entity_count] = entity
                     entity_ids[entity] = entity_count
                     # create node in the graph
                     print NODE_LABEL % (entity_count, entity)
                     entity_count += 1
               entity_id = entity_ids[entity]
               if previous_entities:
                      # create edge from the previous entity for the dialogue sequence
                      print EDGE_LABEL % (entity_ids[previous_entities[-1]], entity_id, annotated_utterance_id)
                      # add links to the new entity from all entities mentioned previously through KG relations
                      if entity not in previous_entities:
                           paths = get_topk_paths([e for e in previous_entities if e != entity], [entity], k=1)
                           for path in set(paths):
                              hops = path[1:-1].split('-<')
                              nhops = len(hops)
                              if nhops < 4:
                                   # print hops
                                   # create edge
                                   start_node = escape(hops[0])
                                   for hop in hops[1:]:
                                        edge_label, next_node = hop.split('>-')
                                        entity = escape(next_node)
                                          # add new entity to the dictionary
                                        if entity not in entity_ids.keys():
                                              entities[entity_count] = entity
                                              entity_ids[entity] = entity_count
                                              # create node in the graph
                                              print EXTERNAL_NODE % (entity_count, entity)
                                              entity_count += 1
                                        print EDGE_LABEL % (entity_ids[start_node], entity_ids[entity], edge_label.split('/')[-1])
                                        start_node = entity
               previous_entities.append(entity)
    # the set of all entities in the dialogue
    # entity_set = entity_ids.keys()


def test_dialogue2dot_graph():
    dialogue2dot_graph(translate_dialog_to_lists(SAMPLE_DIALOGUE))


if __name__ == '__main__':
     test_dialogue2dot_graph()
