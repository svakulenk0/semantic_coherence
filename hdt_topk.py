# -*- coding: utf-8 -*-
'''
svakulenko
27 Feb 2018

Call shortest path algorithm SPARQL web service
'''

import requests

TOPK_SERVICE = 'http://hdt.communidata.at/dbpedia/query'
SAMPLE_QUERY = '''
                PREFIX ppf: <java:at.ac.wu.arqext.path.>
                PREFIX dbr: <http://dbpedia.org/resource/>
                SELECT ?path 
                WHERE { ?path ppf:topk (dbr:IT_Security dbr:ABNT_NBR_15602 6)}
               '''

def get_topk_paths(query):
    '''
    returns a list of paths
    '''
    paths = requests.get(TOPK_SERVICE, params={'query': query, 'output': 'json'}).json()['results']['bindings']
    return [path['path']['value'] for path in paths]


def test_get_topk_paths():
    print get_topk_paths(query=SAMPLE_QUERY)


if __name__ == '__main__':
    test_get_topk_paths()
