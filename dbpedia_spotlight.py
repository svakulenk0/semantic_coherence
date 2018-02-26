# -*- coding: utf-8 -*-
'''
svakulenko
26 Feb 2018

Access dbpedia spotlight service using python library
'''

import spotlight


SAMPLE_TEXT = "President Obama on Monday will call for a new minimum tax rate for individuals making more than $1 million a year to ensure that they pay at least the same percentage of their earnings as other taxpayers, according to administration officials."
LOCAL_SERVICE = 'http://localhost/rest/annotate'
DEMO_SERVICE = 'http://model.dbpedia-spotlight.org/en/annotate'

text = SAMPLE_TEXT
service_address = DEMO_SERVICE
annotations = spotlight.annotate(service_address,
                                 text,
                                 confidence=0.4, support=20)
print annotations
