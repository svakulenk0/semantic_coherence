# -*- coding: utf-8 -*-
'''
svakulenko
26 Feb 2018

Access dbpedia spotlight service directly without the wrapper-library
'''

import requests

DEMO_WEB_SERVICE = 'http://model.dbpedia-spotlight.org/en/annotate'

resp = requests.get(DEMO_WEB_SERVICE, params={'data-urlencode': 'test'})
print resp
