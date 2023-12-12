import argparse
import json
import os
from typing import List, Union

import ruamel.yaml as yaml
from tqdm import tqdm

from tools import SimCSE


class RelationLinker():

    def __init__(self, config):
        self.retrieval_model = SimCSE(config['embed_model'])

        self.triple_directory = config['relation_triple_directory']
        self.index_path = config['relation_index_file']
        self.txt_path = config['relation_json_file']
        self.topk = config['relation_topk']

        self.name_list = self.load_name_list()
        self.build_index(self.index_path)

    def load_name_list(self):
        print("load name list")
        if os.path.exists(self.txt_path):
            return json.load(open(self.txt_path, 'r', encoding='utf-8'))

        name_list = set()
        for file_name in tqdm(os.listdir(self.triple_directory)):
            relation_part = open(os.path.join(self.triple_directory, file_name))
            for line in relation_part:
                item = line.strip().split('\t')
                if len(item) != 3:
                    continue
                name_list.add(item[1])
        name_list = list(name_list)
        json.dump(name_list, open(self.txt_path, 'w', encoding='utf-8'), indent=0)
        return name_list

    def build_index(self, cache_path):
        print("build index begin")
        self.retrieval_model.build_index(self.name_list, cache_path=cache_path)
        print("build index successfully")

    def link_relation(self, mentions: Union[str, List[str]], top_k=None, cached_path=None):
        if top_k is None:
            top_k = self.topk
        if isinstance(mentions, str):
            if top_k == 0:
                return {mentions: None}
            results = self.retrieval_model.search(mentions, threshold=0, top_k=top_k)
            candidate_relations = [relation_name for relation_name, sim in results]
            return {mentions: candidate_relations}
        else:
            all_results = self.retrieval_model.search(mentions, threshold=0, top_k=top_k)
            ret_results = {}
            for i, results in enumerate(all_results):
                candidate_relations = [relation_name for relation_name, sim in results]
                ret_results[mentions[i]] = candidate_relations
            if cached_path:
                print('save relation in cache')
                json.dump(ret_results, open(cached_path, 'w', encoding='utf-8'))
            return ret_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Link.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    linker = RelationLinker(config)

    print(linker.link_relation('measurement_unit.fuel_economy_unit.economy_in_litres_per_kilometre'))
