import argparse
import os.path
import pickle
import urllib

import ruamel.yaml as yaml
from SPARQLWrapper import SPARQLWrapper, JSON

parser = argparse.ArgumentParser()
parser.add_argument('--data_config')
args = parser.parse_args()
data_config = yaml.load(open(args.data_config, 'r'), Loader=yaml.Loader)

sparql = SPARQLWrapper(data_config['sparql_url'])
sparql.setReturnFormat(JSON)

if os.path.exists(data_config['database_cache']):
    f = open(data_config['database_cache'], 'rb')
    database_cache = pickle.load(f)
    print("The cache of the database has been loaded!")
else:
    database_cache = {}

timeout_cache_path = data_config['database_cache'] + '.timeout'
if os.path.exists(timeout_cache_path):
    f = open(timeout_cache_path, 'rb')
    timeout_cache = pickle.load(f)
    print("The cache of the time out has been loaded!")
else:
    timeout_cache = set()


def save_cache():
    print('saving database cache')
    f = open(data_config['database_cache'], 'wb')
    pickle.dump(database_cache, f)
    f.close()

    print('saving timeout cache')
    f = open(timeout_cache_path, 'wb')
    pickle.dump(timeout_cache, f)
    f.close()

    print('save over')


def query2freebase(sparql_txt, query_target):
    if sparql_txt in database_cache:
        return database_cache[sparql_txt]
    if sparql_txt in timeout_cache:
        return []

    try:
        rtn = []
        sparql.setQuery(sparql_txt)
        results = sparql.query().convert()
        if 'COUNT' in sparql_txt:
            query_target = 'callret-0'
        for result in results['results']['bindings']:
            value = result[query_target]['value'].replace('http://rdf.freebase.com/ns/', '')
            value = value.replace('-08:00', '')
            rtn.append(value)
        database_cache[sparql_txt] = rtn
        return rtn
    except urllib.error.URLError as e:
        print("Connection to freebase failed!")
        save_cache()
        exit()
    except Exception as e:
        if 'Transaction timed out' in str(e):
            timeout_cache.add(sparql_txt)
        return []
