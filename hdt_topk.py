# -*- coding: utf-8 -*-
'''
svakulenko
27 Feb 2018

Call shortest path algorithm SPARQL web service

Sample SPARQL query: 
    PREFIX ppf: <java:at.ac.wu.arqext.path.>
    PREFIX dbr: <http://dbpedia.org/resource/>
    SELECT ?path 
    WHERE { ?path ppf:topk (dbr:IT_Security dbr:ABNT_NBR_15602 6)}
'''

import requests

TOPK_SERVICE = 'http://hdt.communidata.at/dbpedia/query'

QUERY_TEMPLATE = '''
                PREFIX ppf: <java:at.ac.wu.arqext.path.>
                PREFIX dbr: <http://dbpedia.org/resource/>
                SELECT ?path 
                WHERE { ?path ppf:topk (dbr:%s dbr:%s %d)}
                 '''

SAMPLE_ENTITIES = [['IT_Security'], ['ABNT_NBR_15602']]

def get_topk_paths(uttered_entities, response_entities, k=6):
    '''
    takes 2 lists of entities: uttered_entities and response_entities
    returns a list of paths
    '''
    query = QUERY_TEMPLATE % (uttered_entities[0], response_entities[0], k)
    try:
        response = requests.get(TOPK_SERVICE, params={'query': query, 'output': 'json'})
        paths = response.json()['results']['bindings']
        return [path['path']['value'] for path in paths]
    except:
        print response.text


def test_get_topk_paths():
    print get_topk_paths(SAMPLE_ENTITIES[0], SAMPLE_ENTITIES[1])


if __name__ == '__main__':
    test_get_topk_paths()
