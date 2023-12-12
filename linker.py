import argparse
import json
import os.path
import pickle
import re
import sys
from collections import Counter

import jsonlines
import openpyxl
import ruamel.yaml as yaml
from tqdm import tqdm

from utils_borrow import s_expression_to_sparql, SemanticMatcher, process_ontology
from database import query2freebase, save_cache
from dataset import DataSet
from entity_linker import EntityLinker
from relation_linker import RelationLinker
from code4instruct import START, JOIN, AND, ARG, CMP, COUNT, STOP

VALUE_LIST = ['#date', '#gyearmonth', '#gyear', '#float', '#integer', '#int', '#double']
VALUE_DICT = {k: k for k in VALUE_LIST}
VALUE_DICT['#gyearmonth'] = '#gYearMonth'
VALUE_DICT['#gyear'] = '#gYear'


class Linker:

    def __init__(self, link_config, data_config):
        self.config = link_config
        self.entity_linker = None
        self.relation_linker = None
        self.matcher = None
        if 'grailqa' in data_config['test_file']:
            self.init_matcher(link_config['fb_roles'], link_config['fb_types'], link_config['reverse_properties'])

    def init_matcher(self, fb_roles, fb_types, reverse_properties):
        reverse_properties, relation_dr, relations, upper_types, types = process_ontology(fb_roles, fb_types,
                                                                                          reverse_properties)
        self.matcher = SemanticMatcher(reverse_properties, relation_dr, relations, upper_types, types)
        print("The matcher has been instanced")

    def is_value(self, match_str):
        new_value_str = None
        for value_trigger in VALUE_LIST:
            if value_trigger in match_str:
                new_value_str = match_str.replace(value_trigger, "") + \
                                '^^http://www.w3.org/2001/XMLSchema' + VALUE_DICT[value_trigger]

                break
        return new_value_str

    @staticmethod
    def clear_arg(query_expression: str):
        query_expression = "(AND" + query_expression[7:]
        query_expression = query_expression.replace("?r", "(JOIN ?r ?z)")
        return query_expression

    def get_relation(self, query_expression: str):
        if query_expression.startswith("(ARGMIN") or query_expression.startswith("(ARGMAX"):
            query_expression = self.clear_arg(query_expression)
        # backward
        try:
            sparql = s_expression_to_sparql(query_expression)
        except Exception:
            print("line: {}\t {}".format(str(sys._getframe().f_lineno), query_expression))
            return [], []
        sparql = sparql.replace('ns:?r', '?r').replace('SELECT DISTINCT ?x', 'SELECT DISTINCT ?r')
        backward_results = set(query2freebase(sparql, 'r'))
        # forward
        query_expression = query_expression.replace('?r', '(R ?r)')
        try:
            sparql = s_expression_to_sparql(query_expression)
        except Exception:
            return [], backward_results
        sparql = sparql.replace('ns:?r', '?r').replace('SELECT DISTINCT ?x', 'SELECT DISTINCT ?r')
        forward_results = set(query2freebase(sparql, 'r'))
        return forward_results, backward_results

    @staticmethod
    def write_excel_xlsx(path, sheet_name, value):
        index = len(value)
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = sheet_name
        for i in range(0, index):
            for j in range(0, len(value[i])):
                sheet.cell(row=i + 1, column=j + 1, value=str(value[i][j]))
        workbook.save(path)
        print("Write data to xlsx file successfully!")

    @staticmethod
    def cal_f1_score(pred_answers, correct_answers):
        hit_answers = pred_answers & correct_answers
        if len(hit_answers) != 0:
            prec = len(hit_answers) / len(pred_answers)
            recall = len(hit_answers) / len(correct_answers)
            f1 = (2 * prec * recall) / (prec + recall)
        else:
            f1 = 0
        return f1

    def cal_em_score(self, pred_expression, correct_expression):
        try:
            em = int(self.matcher.same_logical_form(pred_expression, correct_expression))
        except Exception:
            print("some error happened in exactly matching: {}".format(pred_expression))
            return 0
        return em

    def parse4datasets(self, dataset, data_config):
        START_m = "expression\d? = START\(['\"](.+)['\"]\)"
        JOIN_m = "expression\d? = JOIN\(['\"](.+)['\"], expression\d?\)"
        AND1_m = "expression\d? = AND\(expression\d?, expression\d?\)"
        AND2_m = "expression\d? = AND\(['\"](.+)['\"], expression\d?\)"
        ARG_m = "expression\d? = ARG\(['\"](.+)['\"], expression\d?, ['\"](.+)['\"]\)"
        CMP2_m = "expression\d? = CMP\(['\"](.+)['\"], ['\"](.+)['\"], ['\"]?(.+)['\"]?\)"
        COUNT_m = "expression = COUNT\(expression\)"
        STOP_m = "expression = STOP\(expression\)"
        MATCH_DICT = {'START': START_m, 'JOIN': JOIN_m, 'AND(expression': AND1_m, 'AND': AND2_m,
                      'ARG': ARG_m, 'CMP': CMP2_m, 'COUNT': COUNT_m, 'STOP': STOP_m}
        function_map = {'le': '<=', 'ge': '>=', 'lt': '<', 'gt': '>'}
        CMP_DICT = {v: k for k, v in function_map.items()}
        ENT_m = re.compile("['\"]([gm]\..+)['\"]")
        for trigger in MATCH_DICT.keys():
            MATCH_DICT[trigger] = re.compile(MATCH_DICT[trigger])
        q2a_dict = {}
        completions_generate = data_config['completions_generate']

        with open(data_config['gpt_answer_file'], "r+", encoding="utf-8") as f:
            for i, item in enumerate(jsonlines.Reader(f)):
                user = item[0]['user']
                answers = []
                try:
                    if 'chat' not in item[1]['object']:
                        for choice in item[1]['choices']:
                            if choice['finish_reason'] == 'stop':
                                answers.append(choice['text'])
                            else:
                                answers.append("")
                    else:
                        for choice in item[1]['choices']:
                            if choice['finish_reason'] == 'stop':
                                answers.append(choice['message']['content'])
                            else:
                                answers.append("")
                    assert len(answers) == completions_generate
                    for j, answer in enumerate(answers):
                        q2a_dict[str(int(user) * completions_generate + j)] = answer
                except:
                    print(f'Bad case in gpt answer file, user: {user}')
                    exit(0)
        instance_num = len(q2a_dict.items())
        bad_case_set = set()
        entity_link_batch = []
        relation_link_batch = []
        for index in range(instance_num):
            # preprocess
            function_str = q2a_dict[str(index)]
            function_str = function_str.strip().split("\n\n")[0]
            function_list = function_str.split('\n')
            try:
                if function_list[0] == "'''":
                    function_list = function_list[1:]
                    function_list = function_list[function_list.index("'''") + 1:]
                assert len(function_list) != 0 and len(function_list[0]) != 0
            except:
                bad_case_set.add(index)
                function_list = []
            abs_ent_link_info = []
            entity_link_info = []
            relation_link_info = []

            for i, function in enumerate(function_list[:-1]):
                flag = False
                for trigger in MATCH_DICT.keys():
                    if trigger in function:
                        try:
                            match_results = MATCH_DICT[trigger].match(function).groups()
                            flag = True
                        except Exception:
                            break
                        match_results = list(match_results)
                        if trigger == 'START':
                            if i == 0 and "expression =" not in function:
                                flag = False
                            new_value_str = self.is_value(match_results[0])
                            if new_value_str is not None:
                                function_list[i] = function.replace(match_results[0], new_value_str)
                            elif '.' in match_results[0] and ' ' not in match_results[0] and match_results[0].islower():
                                abs_ent_link_info.append((i, match_results[0]))
                            else:
                                entity_link_info.append((i, match_results[0]))
                        elif trigger == 'JOIN':
                            relation_link_info.append((i, match_results[0]))
                        elif trigger == 'AND':
                            abs_ent_link_info.append((i, match_results[0]))
                        elif trigger == 'ARG':
                            relation_link_info.append((i, match_results[1]))
                            function_list[i] = function.replace(match_results[0], match_results[0].upper())
                        elif trigger == 'CMP':
                            new_value_str = self.is_value(match_results[2])
                            if new_value_str is not None:
                                function_list[i] = function.replace(match_results[2], new_value_str)
                                relation_link_info.append((i, match_results[1]))
                            elif 'expression' in match_results[2]:
                                relation_link_info.append((i, match_results[1]))
                            else:
                                flag = False
                                break
                            if match_results[0] in CMP_DICT.keys():
                                if match_results[0] == '>' and i != 0 and 'START' in function_list[i - 1]:
                                    flag = False
                                function_list[i] = function.replace(match_results[0], CMP_DICT[match_results[0]])
                            elif match_results[0] == '=':
                                function_list[i] = function.replace('CMP', 'JOIN').replace("'=', ", "")
                            elif match_results[0] == '==':
                                function_list[i] = function.replace('CMP', 'JOIN').replace("'==', ", "")
                            else:
                                flag = False
                        break
                if not flag:
                    bad_case_set.add(index)
                    entity_link_info = []
                    abs_ent_link_info = []
                    relation_link_info = []
                    break
            entity_link_batch.append([function_list, entity_link_info, abs_ent_link_info])
            relation_link_batch.append(relation_link_info)

        print(data_config['final_answer_file'])
        final_answer_cache = data_config['final_answer_file'] + '.cache'
        if os.path.exists(final_answer_cache):
            print('Load existing data from final answer cache')
            xlsx_output = pickle.load(open(final_answer_cache, 'rb'))
        else:
            xlsx_output = []

        if not os.path.exists(data_config['entity_result_file']):
            if self.entity_linker is None:
                self.entity_linker = EntityLinker(self.config)
            self.entity_linker.link_entity_batch(entity_link_batch, cache_path=data_config['entity_result_file'])
        print('Entity link finished!')

        if not os.path.exists(data_config['relation_result_file']):
            relation_mention_list = []
            for relation_link in relation_link_batch:
                for (index, relation_mention) in relation_link:
                    relation_mention_list.append(relation_mention)
            if self.relation_linker is None:
                self.relation_linker = RelationLinker(self.config)
            searched_relations_list = self.relation_linker.link_relation(relation_mention_list, cached_path=data_config[
                'relation_result_file'])
        else:
            searched_relations_list = json.load(open(data_config['relation_result_file'], 'r', encoding='utf-8'))

        print('Relation link finished!')
        four_ent_cand = 0
        items = json.load(open(data_config['entity_result_file'], "r+", encoding="utf-8"))
        for index, ((function_list_candidates, _, _), relation_link) in enumerate(zip(items, relation_link_batch)):
            if index < len(xlsx_output):
                continue
            print(f'index: {index}', end='\t')

            if index in bad_case_set:
                print(0, end='\t')
                print("pass the bad case")
                xlsx_output.append(
                    [dataset.questions[index // completions_generate], entity_link_batch[index][0], '', '', '',
                     'pass the bad case', ','.join(dataset.answers[index // completions_generate]), 1, 1, 0, 0])
                continue

            pred_answers = set()
            query_expression, sparql, new_function_list, ent_flag, rel_flag = '', '', [], "null", "null"
            golden_results = set(test_set.ents[index // completions_generate])
            for function_list in function_list_candidates:
                ent_match_results = set(ENT_m.findall('\n'.join(function_list)))
                if len(golden_results) == 0 or len(ent_match_results & golden_results) != 0:
                    ent_flag = "hit"
            for function_list in tqdm(function_list_candidates):
                if len(function_list_candidates) >= 50625:
                    four_ent_cand += 1
                    break
                new_function_list = function_list
                for (i, relation_mention) in relation_link:
                    query_functions = new_function_list[:i + 1]
                    searched_relations = searched_relations_list[relation_mention]
                    if 'time_macro' in relation_mention and '#date' in query_functions[i - 1] and \
                            any(['.from' in r for r in searched_relations]) and \
                            any(['.to' in r for r in searched_relations]):
                        # only for WebQ
                        query_functions = new_function_list[:i + 2]
                        if query_functions[-1] == "expression = AND(expression, expression1)":
                            query_functions[-1] = "expression = AND(expression1, expression)"
                            new_function_list[i + 1] = "expression = AND(expression1, expression)"
                        try:
                            local = {}
                            exec('\n'.join(query_functions), globals(), local)
                            temp_expression = local['expression']
                        except:
                            break
                        try:
                            temp_sparql = s_expression_to_sparql(temp_expression)
                            temp_results = query2freebase(temp_sparql, 'x')
                        except:
                            temp_results = []
                        if len(temp_results) != 0:
                            continue

                    try:
                        local = {}
                        exec('\n'.join(query_functions), globals(), local)
                        query_expression = local['expression']
                        sparql = s_expression_to_sparql(query_expression)
                    except:
                        break
                    query_functions[i] = query_functions[i].replace(relation_mention, "?r")

                    try:
                        local = {}
                        exec('\n'.join(query_functions), globals(), local)
                        query_target = query_functions[i][:query_functions[i].find('=') - 1]
                        query_expression = local[query_target]
                    except:
                        break
                    try:
                        forward_results, backward_results = self.get_relation(query_expression)
                    except:
                        forward_results, backward_results = [], []

                    rel_flag = "null"
                    for relation in searched_relations:
                        if relation in forward_results:
                            rel_flag = "forward"
                            break
                        if relation in backward_results:
                            rel_flag = "backward"
                            break
                    if rel_flag == "null":
                        break
                    elif rel_flag == "forward":
                        new_function_list[i] = new_function_list[i].replace(relation_mention, "(R " + relation + ")")
                    else:
                        new_function_list[i] = new_function_list[i].replace(relation_mention, relation)

                try:
                    local = {}
                    exec('\n'.join(new_function_list), globals(), local)
                    query_expression = local['expression']
                except:
                    print("line: {}\t {}".format(str(sys._getframe().f_lineno), str(new_function_list)))
                    continue
                try:
                    sparql = s_expression_to_sparql(query_expression)
                    pred_answers = set(query2freebase(sparql, 'x'))
                except:
                    print("line: {}\t {}".format(str(sys._getframe().f_lineno), query_expression))
                    continue
                if len(pred_answers) != 0 and list(pred_answers)[0] != '0':
                    break

                # relax the type constraint
                relax_function_list = new_function_list.copy()
                for i in range(2, len(new_function_list)):
                    if any(cons_trigger in new_function_list[-i] for cons_trigger in ["AND", "ARG", "CMP"]):
                        for j in range(len(relax_function_list) - 1, -1, -1):
                            if relax_function_list[j] == new_function_list[-i]:
                                relax_function_list.pop(j)
                        try:
                            local = {}
                            exec('\n'.join(relax_function_list), globals(), local)
                            query_expression = local['expression']
                        except:
                            print("line: {}\t {}".format(str(sys._getframe().f_lineno), '\n'.join(relax_function_list)))
                            break
                        try:
                            sparql = s_expression_to_sparql(query_expression)
                        except:
                            print("line: {}\t {}".format(str(sys._getframe().f_lineno), query_expression))
                            continue
                        pred_answers = set(query2freebase(sparql, 'x'))
                        if len(pred_answers) != 0 and list(pred_answers)[0] != '0':
                            break
                if len(pred_answers) != 0 and list(pred_answers)[0] != '0':
                    break

            correct_answers = set(dataset.answers[index // completions_generate])
            f1 = self.cal_f1_score(pred_answers, correct_answers)
            em = 0
            if test_set.level is not None:
                em = self.cal_em_score(query_expression, test_set.s_expressions[index // completions_generate])
                print("{}\t{}".format(em, f1))
            else:
                print("{}".format(f1))
            xlsx_output.append(
                [dataset.questions[index // completions_generate], entity_link_batch[index][0], new_function_list,
                 query_expression, sparql, ','.join(pred_answers), ','.join(correct_answers),
                 int(ent_flag == 'null'), int(rel_flag == 'null'), em, f1])

            if (index + 1) % 500 == 0:
                print('save xlsx output to cache')
                pickle.dump(xlsx_output, open(final_answer_cache, 'wb'))

        if len(items) % 500 != 0:
            print('save xlsx output to cache')
            pickle.dump(xlsx_output, open(final_answer_cache, 'wb'))
        self.cal_metric(xlsx_output, completions_generate, instance_num, test_set, bad_case_set, four_ent_cand)
        self.cal_metric(xlsx_output, completions_generate, instance_num, test_set, bad_case_set, four_ent_cand,
                        self_consist=False)

    def cal_metric(self, xlsx_output, completions_generate, instance_num, test_set, bad_case_set, four_ent_cand,
                   self_consist=True):
        if test_set.level is not None:
            LEVEL_LIST = ['i.i.d.', 'compositional', 'zero-shot']
            level_f1 = {k: 0 for k in LEVEL_LIST}
            level_em = {k: 0 for k in LEVEL_LIST}
            level_instance_num = {k: 0 for k in LEVEL_LIST}
            for index in range(instance_num // completions_generate):
                level_instance_num[test_set.level[index]] += 1
        xlsx_final_output = [['question', 'original function', 'replaced function', 's-expression',
                              'sparql', 'pre_answers', 'golden_answers', 'ent_no_hit', 'rel_no_hit', 'exact match',
                              'f1 score']]
        avg_em, avg_f1, ent_no_hit, rel_no_hit = 0, 0, 0, 0
        bad_cnt = 0
        for index in range(0, len(xlsx_output), completions_generate):
            answers = xlsx_output[index: index + completions_generate]
            if not self_consist:
                answers = answers[0:1]
            no_hit_list = [item[-4] for item in answers]
            ent_no_hit_flag = all([item == 1 for item in no_hit_list])
            no_hit_list = [item[-3] for item in answers]
            rel_no_hit_flag = all([item == 1 for item in no_hit_list])
            if ent_no_hit_flag:
                ent_no_hit += 1
            if rel_no_hit_flag:
                rel_no_hit += 1
            answer_to_data = {}
            answer_candidate = []
            for answer in answers:
                pred_ans = answer[5]
                if pred_ans and pred_ans != 'pass the bad case':
                    answer_to_data[tuple(pred_ans)] = answer
                    answer_candidate.append(tuple(pred_ans))
            if all([idx in bad_case_set for idx in range(index, index + len(answers))]):
                bad_cnt += 1
            if not answer_to_data:
                answer = answers[0]
                xlsx_final_output.append(answer)
            else:
                count_dict = Counter(answer_candidate)
                pred_ans = max(count_dict, key=count_dict.get)
                xlsx_final_output.append(answer_to_data[pred_ans])
                avg_f1 += answer_to_data[pred_ans][-1]
                avg_em += answer_to_data[pred_ans][-2]
                if test_set.level is not None:
                    level_f1[test_set.level[index // completions_generate]] += answer_to_data[pred_ans][-1]
                    level_em[test_set.level[index // completions_generate]] += answer_to_data[pred_ans][-2]

        if self_consist:
            print('=================SC================')
        else:
            print('===============No SC===============')
        print(f'Format Error Rate(FER): {bad_cnt / (instance_num // completions_generate) * 100}')
        print("F1 Score: {}".format(avg_f1 / (instance_num // completions_generate) * 100))
        if test_set.level is not None:
            for key in level_f1.keys():
                print("\tF1 Score for {}: {}".format(key, level_f1[key] / level_instance_num[key] * 100))
            print("EM Score: {}".format(avg_em / (instance_num // completions_generate) * 100))
            for key in level_em.keys():
                print("EM Score for {}: {}".format(key, level_em[key] / level_instance_num[key] * 100))
        print('Entity miss count: {}, relation miss count: {}'.format(ent_no_hit, rel_no_hit))
        print('More than 4 entity candidates: {}'.format(four_ent_cand))
        if self_consist:
            book_name_xlsx = data_config['final_answer_file']
            self.write_excel_xlsx(book_name_xlsx, 'results', xlsx_final_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config')
    parser.add_argument('--link_config', default='./configs/Link.yaml')
    args = parser.parse_args()
    data_config = yaml.load(open(args.data_config, 'r'), Loader=yaml.Loader)
    link_config = yaml.load(open(args.link_config, 'r'), Loader=yaml.Loader)

    test_set = DataSet(data_config, type='test')
    linker = Linker(link_config, data_config)
    linker.parse4datasets(test_set, data_config)
    save_cache()
