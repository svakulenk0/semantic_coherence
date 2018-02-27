# -*- coding: utf-8 -*-
'''
svakulenko
27 Feb 2018

Explains dialogue coherence relations between utterances using background semantic knowledge from the DBpedia graph
'''
from dbpedia_spotlight import annotate_entities
from hdt_topk import get_topk_paths

SAMPLE_COHERENT_DIALOGUE = ["Have you seen Lady Bird", "Greta Gerwig is a talented director",
                            "David Fincher is my absolute favourite"]
SAMPLE_UNCOHERENT_DIALOGUE = ["Have you seen Lady Bird", "David Fincher is my absolute favourite"]


def trace_dialogue_semantics(dialogue):
    uttered_entities = annotate_entities(dialogue[0])
    for response in dialogue[1:]:
        response_entities = annotate_entities(response)
        print uttered_entities, response_entities
        print get_topk_paths(uttered_entities, response_entities)
        uttered_entities = response_entities


def test_trace_dialogue_semantics():
    print("\nCoherent dialogue:")
    trace_dialogue_semantics(SAMPLE_COHERENT_DIALOGUE)


def show_uncoherent_dialogue_relations():
    print("\nUncoherent dialogue:")
    trace_dialogue_semantics(SAMPLE_UNCOHERENT_DIALOGUE)


if __name__ == '__main__':
    test_trace_dialogue_semantics()
    show_uncoherent_dialogue_relations()
