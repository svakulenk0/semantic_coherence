# -*- coding: utf-8 -*-
'''
svakulenko
27 Feb 2018

Call shortest path algorithm SPARQL web service

Sample SPARQL queries: 
    
    PREFIX : <http://dbpedia.org/resource/>
    PREFIX ppf: <java:at.ac.wu.arqext.path.>
    SELECT * WHERE {
    ?X ppf:topk ("--source" :Alot :EMachines "--target" :High-definition_television :CPU_cache "--k" 10 "--maxlength" 25)
    }

    PREFIX ppf: <java:at.ac.wu.arqext.path.>
    PREFIX dbr: <http://dbpedia.org/resource/>
    SELECT ?path 
    WHERE { ?path ppf:topk (dbr:IT_Security dbr:ABNT_NBR_15602 6)}

    PREFIX ppf: <java:at.ac.wu.arqext.path.>
    PREFIX dbr: <http://dbpedia.org/resource/>
    SELECT ?path
    WHERE { ?path ppf:topk (?X ?Y 2) . 
    VALUES (?X ) { (dbr:Felipe_Massa)(dbr:1952_Winter_Olympics) } .
    VALUES (?Y ) { (dbr:Elliot_Richardson) (dbr:Red_Bull) }  }

    PREFIX ppf: <java:at.ac.wu.arqext.path.>
    PREFIX dbr: <http://dbpedia.org/resource/>
    SELECT ?path
    WHERE { ?path ppf:topk (?X ?Y 2) . 
    VALUES (?X ) { (dbr:Arch)(dbr:Organisation_of_Islamic_Cooperation) } .
    VALUES (?Y ) { (dbr:Sudo) (dbr:CPU_cache) }  }

'''

import requests
import random
import signal

TOPK_SERVICE = 'http://svhdt.ai.wu.ac.at/dbpedia/query'
DBPEDIA_ENDPOINT = 'http://dbpedia.org/sparql'

QUERY_TEMPLATE = '''
                PREFIX ppf: <java:at.ac.wu.arqext.path.>
                PREFIX dbr: <http://dbpedia.org/resource/>
                SELECT * WHERE {
                ?X ppf:topk ("--source" %s "--target" %s "--k" %d "--maxlength" %d "--timeout" 100)
                }
                 '''

RANDOM_CONCEPT = '''
                SELECT ?s WHERE { ?s ?p ?o }
                OFFSET %d
                LIMIT 1
                '''

ENTITY_TEMPLATE = "dbr:%s"

SAMPLE_ENTITIES = [['IT_Security'], ['ABNT_NBR_15602']]
SAMPLE_ENTITIES2 = [['Lady_Bird_Johnson'], ['Greta_Gerwig']]
SAMPLE_ENTITIES3 = [['Lady_Bird_Johnson', 'CPU_cache'], ['Greta_Gerwig', 'EMachines']]


def handler(signum, frame):
    raise Exception("time out!")


def get_random():
    try:
        query = RANDOM_CONCEPT % random.randint(0, 4000000)
        response = requests.get(DBPEDIA_ENDPOINT, params={'query': query, 'output': 'json'})
        return response.json()['results']['bindings'][0]['s']['value'].split('/')[-1]
    except:
        print response.text


def get_topk_paths(uttered_entities, response_entities, k=10, max_length=25):
    '''
    takes 2 lists of entities: uttered_entities and response_entities
    returns a list of paths
    '''
    query = QUERY_TEMPLATE % (' '.join([ENTITY_TEMPLATE % e for e in uttered_entities]),
                              ' '.join([ENTITY_TEMPLATE % e for e in response_entities]),
                              k, max_length)
    # print query
    # signal.signal(signal.SIGALRM, handler)
    # signal.alarm(60)
    try:
        response = requests.get(TOPK_SERVICE, params={'query': query, 'output': 'json'})
        paths = response.json()['results']['bindings']
        return [path['X']['value'] for path in paths]
    except Exception, exc:
        print exc
        # produce an empty path on time out
        return []


def test_get_random():
    print get_random()


def test_get_topk_paths():
    k = 10
    paths = get_topk_paths(SAMPLE_ENTITIES3[0], SAMPLE_ENTITIES3[1], k=k)
    assert len(paths) == k
    print paths


def random_shortest_paths():
    '''
    estimate length of the shortest path between 2 random entities
    '''
    k = 1
    concept1 = get_random()
    concept2 = get_random()
    print concept1, concept2
    paths = get_topk_paths([concept1], [concept2], k=k, max_length=250000)
    print paths
    if paths:
        print len(paths[0])


if __name__ == '__main__':
    print random_shortest_paths()
    # test_get_topk_paths()
