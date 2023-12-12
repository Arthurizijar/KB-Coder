import json
import random

import networkx as nx
from networkx.readwrite import json_graph


class DataSet:

    def __init__(self, config, type):
        self.embedding_model = config['embed_model']
        self.sample_type = config['sample_type']
        self.k = config['k']
        self.random_seed = config['seed'] if 'seed' in config else 42
        self.level, self.ents, self.s_expressions = None, None, None
        if type == "train":
            self.file_path = config['train_file']
            self.questions, self.graphs, self.answers = self.load_data(self.file_path)
            self.build_sample_pool(config['cache_file'])
        else:
            self.file_path = config['test_file']
            self.questions, self.graphs, self.answers = self.load_data(self.file_path)
        self.actions, self.visited_ents, self.visited_edges, self.target_vars = self.prepare_action()

    def build_sample_pool(self, cache_path):
        if "simcse" in self.embedding_model:
            from tools import SimCSE
            self.retrieval_model = SimCSE(self.embedding_model)
            self.retrieval_model.build_index(self.questions, cache_path=cache_path)
            print("build index successfully")
            self.q2i = {q: i for i, q in enumerate(self.questions)}
        else:
            raise NotImplementedError

    def sample_demos(self, questions):
        ret_results = []
        if self.sample_type == 'topk':
            all_results = self.retrieval_model.search(questions, threshold=0, top_k=self.k)
            for j in range(len(questions)):
                temp = []
                for q, sim in all_results[j]:
                    idx = self.q2i[q]
                    temp.append(idx)
                ret_results.append(temp)
            return ret_results
        elif self.sample_type == 'slice_random':
            random.seed(self.random_seed)
            assert len(self.questions) >= self.k
            slice_size = len(self.questions) // self.k
            remain_cnt = len(self.questions) % self.k
            sample_demos_idxs = []
            start = 0
            for i in range(self.k):
                end = start + slice_size + [0, 1][i < remain_cnt]
                sample_demos_idxs.append(random.randint(start, end - 1))
                start = end
            sample_demos_idxs = tuple(sample_demos_idxs)
            for j in range(len(questions)):
                ret_results.append(sample_demos_idxs)
            return ret_results
        else:
            raise NotImplementedError

    def load_data(self, path):
        questions, graphs, answers, ents = [], [], [], []
        if "WQ" in path:
            with open(path, 'r', encoding='utf8') as fp:
                json_data = json.load(fp)
                for item in json_data:
                    question = item['ProcessedQuestion']
                    flag = False
                    for i in range(len(item['Parses'])):
                        if item['Parses'][i]['s_expression'] != "null":
                            flag = True
                            graph = item['Parses'][0]['graph_query']
                            answer = item['Parses'][0]['Answers']
                            answer = [item['AnswerArgument'] for item in answer]
                    if flag:
                        questions.append(question)
                        graphs.append(graph)
                        answers.append(answer)
                        if 'test' in path:
                            ents.append([temp['mid'] for temp in graph['nodes'] if temp['node_type'] == 'entity'])
            self.ents = ents
        elif 'GrailQA' in path:
            level, ents, s_expressions = [], [], []
            with open(path, 'r', encoding='utf-8') as fp:
                json_data = json.load(fp)
                for item in json_data:
                    item['graph_query']['directed'] = False
                    questions.append(item['question'])
                    graphs.append(item['graph_query'])
                    answer = [ans_dict['answer_argument'] for ans_dict in item['answer']]
                    answers.append(answer)
                    if 'dev' in path:
                        ents.append(
                            [temp['mid'] for temp in item['graph_query']['nodes'] if temp['node_type'] == 'entity'])
                        level.append(item['level'])
                        s_expressions.append(item['s_expression'])
            self.level = level
            self.ents = ents
            self.s_expressions = s_expressions
        elif 'GraphQ' in path:
            ents = []
            with open(path, 'r', encoding='utf-8') as fp:
                json_data = json.load(fp)
                for item in json_data:
                    item['graph_query']['directed'] = False
                    questions.append(item['question'])
                    graphs.append(item['graph_query'])
                    answer = [ans_dict['answer_argument'] for ans_dict in item['answer']]
                    answers.append(answer)
                    if 'test' in path:
                        ents.append(
                            [temp['mid'] for temp in item['graph_query']['nodes'] if temp['node_type'] == 'entity'])
            self.ents = ents
        return questions, graphs, answers

    def prepare_action(self):
        actions_list, visited_ent_list, visited_edge_list, target_var_list = [], [], [], []
        bad_case_questions = []
        for j, (question, graph) in enumerate(zip(self.questions, self.graphs)):
            graph['directed'] = False
            g = json_graph.node_link_graph(graph)
            node_num, nodes, edges = len(g.nodes), g.nodes, g.edges
            topic_ent, question_ent = 0, 0
            triples, actions = [], []
            for i in range(1, node_num + 1):
                if nodes[i]['question_node'] == 1:
                    question_ent = i
            # find the entity which is farthest to the question node to be the topic entity
            length = dict(nx.all_pairs_shortest_path_length(g))
            longest_length = 0
            for i in range(1, node_num + 1):
                if nodes[i]['node_type'] == 'entity':
                    if longest_length < length[question_ent][i]:
                        topic_ent = i
                        longest_length = length[question_ent][i]
            if topic_ent == 0:
                for i in range(1, node_num + 1):
                    if nodes[i]['node_type'] == 'literal':
                        topic_ent = i
                        break
            assert topic_ent != 0
            current_ent = topic_ent
            visited_ent, visited_edge, target_var = list(), list(), list()
            visited_ent.append(current_ent)
            target_var.append(current_ent)
            bad_case_flag = False
            while len(visited_ent) < node_num:
                next_nodes = g.adj[current_ent]
                if nodes[current_ent]['node_type'] in ['entity', 'literal']:
                    assert len(next_nodes) == 1
                    actions.append("JOIN")
                    last_ent = current_ent
                    current_ent = list(next_nodes.keys())[0]
                    visited_ent.append(current_ent)
                    visited_edge.append((last_ent, current_ent)) if (last_ent, current_ent) in edges \
                        else visited_edge.append((current_ent, last_ent))
                elif nodes[current_ent]['node_type'] == 'class':
                    flag = False
                    for id in next_nodes.keys():
                        if nodes[id]['node_type'] == 'entity' and id not in visited_ent:
                            actions.append("AND")
                            visited_ent.append(id)
                            target_var.append(id)
                            visited_edge.append((id, current_ent)) if (id, current_ent) in edges \
                                else visited_edge.append((current_ent, id))
                            flag = True
                            break
                        if nodes[id]['node_type'] == 'literal' and id not in visited_ent:
                            if nodes[id]['function'] in ['count', 'argmax', 'argmin']:
                                actions.append("ARG")
                            elif nodes[id]['function'] in ['>', '<', '>=', '<=']:
                                actions.append("CMP")
                            elif nodes[id]['function'] in ['none']:
                                actions.append("AND")
                            visited_ent.append(id)
                            target_var.append(id)
                            visited_edge.append((id, current_ent)) if (id, current_ent) in edges \
                                else visited_edge.append((current_ent, id))
                            flag = True
                            break
                    if flag:
                        continue
                    next_hop_node = None
                    remained_nodes = [id for id in next_nodes.keys() if
                                      nodes[id]['node_type'] == 'class' and id not in visited_ent]
                    remained_nodes_length = {id: length[id][question_ent] for id in remained_nodes}
                    remained_nodes = sorted(remained_nodes_length, reverse=True)
                    if current_ent != question_ent:
                        remained_nodes, next_hop_node = remained_nodes[:-1], remained_nodes[0]
                    for id in remained_nodes:
                        constraint_node = [k for k in g.adj[id] if k != current_ent]
                        if len(constraint_node) != 1:
                            bad_case_flag = True
                            break
                        constraint_node = constraint_node[0]
                        assert nodes[constraint_node]['node_type'] in ['entity', 'literal']
                        visited_edge.append((constraint_node, id)) if (constraint_node, id) in edges \
                            else visited_edge.append((id, constraint_node))
                        visited_edge.append((id, current_ent) if (id, current_ent) in edges \
                                                else visited_edge.append(current_ent, id))
                        visited_ent += [constraint_node, id]
                        target_var += [constraint_node, id]

                    if bad_case_flag:
                        break
                    target_var.append(current_ent)
                    if next_hop_node is not None:
                        visited_edge.append((next_hop_node, current_ent)) if (next_hop_node, current_ent) in edges \
                            else visited_edge.append((current_ent, next_hop_node))
                        visited_ent.append(current_ent)
                        current_ent = next_hop_node

            actions.append("STOP")
            actions_list.append(actions)
            visited_ent_list.append(visited_ent)
            target_var_list.append(target_var)
            visited_edge_list.append(visited_edge)

        for question in bad_case_questions:
            self.questions.remove(question)
        return actions_list, visited_ent_list, visited_edge_list, target_var_list
