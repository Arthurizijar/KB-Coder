import json

import networkx as nx
from networkx.readwrite import json_graph

from dataset import DataSet


class Selector:

    def __init__(self, config):
        self.description = config['selector_desc']
        self.template = config['selector_temp']
        self.r_cnt = config.get('r_cnt', 1)
        self.index = self.get_index()

    @staticmethod
    def get_index():
        i = 0
        while True:
            yield str(i)
            i += 1

    def queries_gen(self, config, train_set: DataSet, test_set: DataSet, relation_linker, export_answer=True):
        print("There are {} questions in the test set totally".format(len(test_set.questions)))
        questions = test_set.questions

        json_queries = []
        demo_idxs = train_set.sample_demos(questions)

        relation_idxs = relation_linker.link_relation(questions, top_k=self.r_cnt)

        for i, question in enumerate(questions):
            json_query = self.format_query(i, train_set, test_set, 'one_by_one',
                                           demo_idxs[i] if demo_idxs else None,
                                           relation_idx=relation_idxs[question] if relation_idxs else None)
            # json_query = self.format_query(i, train_set, test_set, 'one_by_one', demo_idxs[i], relation_idx=None)
            json_queries += json_query
        with open(config['query_file'], "w") as f:
            for json_query in json_queries:
                json_string = json.dumps(json_query)
                f.write(json_string + "\n")
        if export_answer:
            json_answers = self.answer_gen(test_set, 'one_by_one', json_queries)
            with open(config['golden_answer_file'], "w") as f:
                for json_answer in json_answers:
                    json_string = json.dumps(json_answer)
                    f.write(json_string + "\n")

    def answer_gen(self, test_set: DataSet, mode, json_queries):
        json_answers = []
        actions = []
        if mode == 'one_by_one':
            for action_list in test_set.actions:
                actions += action_list
        elif mode == 'all_in_one':
            for action_list in test_set.actions:
                actions.append(action_list[-1])
        # 可能不测所有数据，因此需要截断
        actions = actions[:len(json_queries)]
        for q, a in zip(json_queries, actions):
            json_format = {
                'query': q['question'],
                'demo': q['demo_examples'],
                'answer': a,
                'user': q['user']
            }
            json_answers.append(json_format)
        return json_answers

    def format_query(self, target_idx, train_set: DataSet, test_set: DataSet, mode, demo_idx=None, relation_idx=None):
        demo_examples = ''
        if demo_idx is not None:
            demo_examples = self.format_demo(demo_idx, train_set)
        # 目前此处仅从数据集中取
        question_example = self.format_question(test_set, target_idx)
        if relation_idx is not None:
            demo_examples += '\n' + self.format_relation_help(relation_idx)

        if mode == 'one_by_one':
            json_list = []
            for item in question_example:
                json_format = {'question': item,
                               'demo_examples': demo_examples,
                               'user': self.index.__next__()}
                # model, temperature and max_tokens are added before post the query
                json_list.append(json_format)
            return json_list
        elif mode == 'all_in_one':
            json_format = {'question': question_example,
                           'demo_examples': demo_examples,
                           'user': self.index.__next__()}
            return json_format
        else:
            raise NotImplementedError

    def format_question(self, test_set, target_idx):
        question, actions = test_set.questions[target_idx], test_set.actions[target_idx]
        return ["question = '{}'".format(question) + '\n']

    def format_demo(self, demo_idx, train_set: DataSet):
        demo_example_list = []
        for i in demo_idx:
            # use function
            graph, visited_edge, target_var = train_set.graphs[i], train_set.visited_edges[i], train_set.target_vars[i]
            graph_list = self.format_function_4_grailqa(graph, visited_edge, target_var)

            question_template = "question = '{}'".format(train_set.questions[i])
            demo_str = question_template + '\n' + '\n'.join(graph_list)
            demo_example_list.append(demo_str)
        return '\n'.join(demo_example_list)

    def format_function_4_grailqa(self, graph, visited_edge, target_var):
        START = "expression{} = START('{}')"
        JOIN = "expression{} = JOIN('{}', {})"
        AND = "expression{} = AND({}, {})"
        ARG = "expression{} = ARG('{}', {}, '{}')"
        CMP = "expression{} = CMP('{}', '{}', {})"
        COUNT = "expression = COUNT(expression)"
        STOP = "expression = STOP(expression)\n"
        g = json_graph.node_link_graph(graph)
        question_ent = self.get_question_ent(g)
        core_path = nx.bidirectional_shortest_path(g, target_var[0], question_ent)
        nodes, edges = g.nodes(), g.edges()
        ex_idx = 1
        function_list = []
        count_flag = False
        for i in range(len(visited_edge)):
            (node1, node2) = visited_edge[i]
            start_idx = target_var[i]
            node_type = nodes[start_idx]['node_type']
            next_idx = node1 if node2 == target_var[i] else node2
            relation = g[node1][node2][0]['label']
            if start_idx in core_path and next_idx in core_path:
                if node_type == 'entity':
                    function_list.append(START.format('', nodes[start_idx]['friendly_name'].lower()))
                    function_list.append(JOIN.format('', relation, 'expression'))
                elif node_type == 'class':
                    function_list.append(JOIN.format('', relation, 'expression'))
                elif node_type == 'literal':
                    function = nodes[start_idx]['function']
                    if function == 'none':
                        function_list.append(START.format('', nodes[start_idx]['mid'].lower().replace(
                            '^^http://www.w3.org/2001/xmlschema', '')))
                        function_list.append(JOIN.format('', relation, 'expression'))
                    elif function in ['>=', '<=', '>', '<']:
                        function_list.append(START.format('', nodes[start_idx]['mid'].lower().replace(
                            '^^http://www.w3.org/2001/xmlschema', '')))
                        function_list.append(CMP.format('', function, relation, 'expression'))
                    elif function in ['argmin', 'argmax']:
                        function_list.append(START.format('', nodes[next_idx]['mid']))
                        function_list.append(ARG.format('', function, 'expression', relation))
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                if nodes[next_idx]['function'] == 'count':
                    count_flag = True
            else:
                if node_type == 'entity':
                    function_list.append(START.format(ex_idx, nodes[start_idx]['friendly_name'].lower()))
                    function_list.append(JOIN.format(ex_idx, relation, 'expression' + str(ex_idx)))
                elif node_type == 'class':
                    function_list.append(JOIN.format(ex_idx, relation, 'expression' + str(ex_idx)))
                elif node_type == 'literal':
                    function = nodes[start_idx]['function']
                    if function == 'none':
                        function_list.append(START.format(ex_idx, nodes[start_idx]['mid'].lower().replace(
                            '^^http://www.w3.org/2001/xmlschema', '')))
                        function_list.append(JOIN.format(ex_idx, relation, 'expression' + str(ex_idx)))
                    elif function in ['>=', '<=', '>', '<']:
                        function_list.append(START.format(ex_idx, nodes[start_idx]['mid'].lower().replace(
                            '^^http://www.w3.org/2001/xmlschema', '')))
                        function_list.append(CMP.format(ex_idx, function, relation, 'expression' + str(ex_idx)))
                    elif function in ['argmin', 'argmax']:
                        if next_idx in core_path:
                            function_list.append(ARG.format('', function, 'expression', relation))
                            continue
                        else:
                            function_list.append(START.format(ex_idx, nodes[next_idx]['mid']))
                            function_list.append(ARG.format(ex_idx, function, 'expression' + str(ex_idx), relation))
                else:
                    raise NotImplementedError
                if (start_idx in core_path and next_idx not in core_path) or (
                        next_idx in core_path and start_idx not in core_path):
                    function_list.append(AND.format('', 'expression', 'expression' + str(ex_idx)))
                    ex_idx += 1
        if count_flag:
            function_list.append(COUNT)
        function_list.append(STOP)
        return function_list

    @staticmethod
    def format_relation_help(relation_list):
        RELATION = "'''\nSome relations for reference are as follows:\n{}\n'''"
        relation_str = RELATION.format('\n'.join(relation_list))
        return relation_str

    @staticmethod
    def get_question_ent(g):
        nodes = g.nodes
        for i in range(1, len(nodes) + 1):
            if nodes[i]['question_node'] == 1:
                return i
