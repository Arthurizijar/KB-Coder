import itertools
import json
import os
import time
from typing import List

from tqdm import tqdm

from tools import SimCSE


class EntityLinker:

    def __init__(self, config):
        # entity
        self.retrieval_model = SimCSE(config['embed_model'])
        self.entity_surface_file = config['entity_surface_name_file']
        self.index_path = config['entity_index_file']
        self.json_path = config['entity_json_file']

        # abs entity
        self.abs_retrieval_model = SimCSE(config['embed_model'])
        self.abs_entity_file = config['abs_entity_name_file']
        self.abs_index_path = config['abs_entity_index_file']
        self.abs_json_path = config['abs_entity_json_file']

        self.name_dict, self.abs_name_list = self.load_data()
        self.build_index()
        self.topk = config['entity_topk']

    def load_data(self):
        if os.path.exists(self.json_path):
            print("load entity data from cache")
            name_dict = json.load(open(self.json_path, 'r', encoding='utf-8'))
        else:
            print("load entity data from entity file")
            name_dict = self.load_entity_data()

        if os.path.exists(self.abs_json_path):
            print("load abs entity data from cache")
            abs_name_list = json.load(open(self.abs_json_path, 'r', encoding='utf-8'))
        else:
            print("load abs entity data from type file")
            abs_name_list = self.load_abs_entity_data()

        return name_dict, abs_name_list

    def load_entity_data(self):
        name_dict = {}
        entity_surface = open(self.entity_surface_file, 'r', encoding='utf-8')
        for idx, line in enumerate(entity_surface.readlines()):
            item = line.strip().split('\t')
            if len(item) != 3:
                continue
            surface_name, popularity, name = item[0].lower(), float(item[1]), item[2]
            name_dict.setdefault(surface_name, []).append((name, popularity))
        for surface_name in name_dict:
            name_dict[surface_name] = sorted(name_dict[surface_name], key=lambda x: x[1], reverse=True)
        json.dump(name_dict, open(self.json_path, 'w', encoding='utf-8'), indent=4)
        return name_dict

    def load_abs_entity_data(self):
        name_set = set()
        abs_entity_file = open(self.abs_entity_file, 'r', encoding='utf-8')
        for idx, line in enumerate(abs_entity_file.readlines()):
            item = line.strip().rstrip('.').strip().split()
            if len(item) != 3:
                continue
            head_type, tail_type = item[0], item[2]
            name_set.add(head_type)
            name_set.add(tail_type)
        name_set = list(name_set)
        json.dump(name_set, open(self.abs_json_path, 'w', encoding='utf-8'), indent=4)
        return name_set

    def build_index(self):
        print("build index begin")
        start_time = time.time()
        self.retrieval_model.build_index(list(self.name_dict.keys()), cache_path=self.index_path)
        self.abs_retrieval_model.build_index(self.abs_name_list, cache_path=self.abs_index_path)
        print(f"build index successfully, took {time.time() - start_time}s")

    def hard_match(self, mention):
        mention = mention.lower().strip()
        if mention in self.name_dict:
            return self.name_dict[mention]
        return None

    def sim_match(self, mentions: List[str]):
        if len(mentions) == 0:
            return []
        all_results = self.retrieval_model.search(mentions, threshold=0, top_k=self.topk)
        ret_results = []
        for results in all_results:
            ret_result = []
            for surface_name, sim in results:
                ret_result.extend(self.name_dict[surface_name])
                if len(ret_result) >= self.topk:
                    break
            ret_results.append(ret_result[:self.topk])
        return ret_results

    def link_abstract_entity(self, mentions):
        if len(mentions) == 0:
            return []
        all_results = self.abs_retrieval_model.search(mentions, threshold=0, top_k=self.topk)
        ret_results = {}
        for i, results in enumerate(all_results):
            candidate_abs = [abs_name for abs_name, sim in results]
            ret_results[mentions[i]] = candidate_abs
        return ret_results

    def link_entity_batch(self, items, *, cache_path=None):
        print('Link entities begins')
        mention_set = set()
        abs_mention_set = set()
        for func_list, index_tuple, abs_index_tuple in items:
            for idx, mention in index_tuple:
                mention_set.add(mention)
            for idx, mention in abs_index_tuple:
                abs_mention_set.add(mention)

        mention2name = {}
        # hard match
        unmatched_mention_set = set()
        for mention in mention_set:
            matched_entities = self.hard_match(mention)
            if matched_entities:
                mention2name[mention] = [ent for ent, _ in matched_entities[:self.topk]]
                if len(mention2name[mention]) < self.topk:
                    unmatched_mention_set.add(mention)
            else:
                unmatched_mention_set.add(mention)

        # similarity match
        unmatched_mention_set = list(unmatched_mention_set)
        all_results = self.sim_match(unmatched_mention_set)
        assert len(unmatched_mention_set) == len(all_results)
        for mention, results in zip(unmatched_mention_set, all_results):
            names = [ent for ent, _ in results[:self.topk]]
            mention2name[mention] = (mention2name.get(mention, []) + names)[:self.topk]

        # abs similarity match
        abs_mention_set = list(abs_mention_set)
        mention2abs_name = self.link_abstract_entity(abs_mention_set)

        # replace
        print('Replacing mentions...')
        for i, (func_list, index_tuple, abs_index_tuple) in enumerate(tqdm(items)):
            candiate_list = []
            for idx, mention in index_tuple:
                candiate_list.append([(idx, mention, name) for name in mention2name[mention]])
            for idx, mention in abs_index_tuple:
                candiate_list.append([(idx, mention, name) for name in mention2abs_name[mention]])

            if len(candiate_list) >= 5:
                items[i][0] = []
                continue
            entity_combinations = list(itertools.product(*candiate_list))

            replace_func_lists = []
            for entity_combination in entity_combinations:
                func_list_copy = func_list.copy()
                for idx, mention, name in entity_combination:
                    func_list_copy[idx] = name.join(func_list_copy[idx].rsplit(mention, 1))
                replace_func_lists.append(func_list_copy)
            items[i][0] = replace_func_lists

        print('Link entity over, save to path.')
        if cache_path:
            json.dump(items, open(cache_path, 'w', encoding='utf-8'), indent=4)
        return items
