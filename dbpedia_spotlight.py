# -*- coding: utf-8 -*-
'''
svakulenko
26 Feb 2018

Access dbpedia spotlight service using python library
'''

import spotlight

SAMPLE_TEXT = "President Obama on Monday will call for a new minimum tax rate for individuals making more than $1 million a year to ensure that they pay at least the same percentage of their earnings as other taxpayers, according to administration officials."
LOCAL_SERVICE = 'http://localhost/rest/annotate'
# http://api.dbpedia-spotlight.org/en
DEMO_SERVICE = 'http://model.dbpedia-spotlight.org/en/annotate'


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


def annotate_entities(text, service_address=DEMO_SERVICE):
    # .split('/')[-1]
    try:
        entities = spotlight.annotate(service_address, text,
                                      confidence=0.4, support=20)
        return entities
        # return list(set([entity['URI'].split('/')[-1] for entity in entities]))
        # return list(set([entity['URI'] for entity in entities]))
    except:
        return None


def test_annotate_entities():
    print annotate_entities(SAMPLE_TEXT)


if __name__ == '__main__':
    test_annotate_entities()
