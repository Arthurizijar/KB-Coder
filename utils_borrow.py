import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import List

import networkx as nx
import ruamel.yaml as yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from SP_tools.components.expr_parser import parse_s_expr

parser = argparse.ArgumentParser()
parser.add_argument('--link_config', default='./configs/Link.yaml')
args = parser.parse_args()
link_config = yaml.load(open(args.link_config, 'r'), Loader=yaml.Loader)

REVERSE = False

current_directory = os.path.dirname(os.path.abspath(__file__))
with open(link_config['fb_roles'], 'r') as f:
    content = f.readlines()
relation_dr = {}
relations = set()
for line in content:
    fields = line.split()
    relation_dr[fields[1]] = (fields[0], fields[2])
    relations.add(fields[1])

reverse_properties = {}
with open(link_config['reverse_properties'], 'r') as f:
    for line in f:
        reverse_properties[line.split('\t')[0]] = line.split('\t')[1].replace('\n', '')

with open(link_config['fb_types'], 'r') as f:
    content = f.readlines()
upper_types = defaultdict(lambda: set())
types = set()
for line in content:
    fields = line.split()
    upper_types[fields[0]].add(fields[2])
    types.add(fields[0])
    types.add(fields[2])

function_map = {'le': '<=', 'ge': '>=', 'lt': '<', 'gt': '>'}
id2name_dict = None


def load_nameid_dict(file_dir, lower):
    print("Loading name2id and id2name dict...")
    name2id_dict = defaultdict(list)
    id2name_dict = {}
    for file in tqdm(os.listdir(file_dir)):
        with open(os.path.join(file_dir, file), 'r') as rf:
            data_input = csv.reader(rf, delimiter="\t")
            for row in data_input:
                if lower:
                    procesed_name = row[2].lower()
                else:
                    procesed_name = row[2]
                name2id_dict[procesed_name].append(row[0])
                id2name_dict[row[0]] = procesed_name
    return name2id_dict, id2name_dict


def logical_form_to_graph(expression: List, friendly_name=True) -> nx.MultiGraph:
    global id2name_dict
    if friendly_name and id2name_dict is None:
        name2id_dict, id2name_dict = load_nameid_dict(link_config['id2name_parts'], lower=False)
    G = _get_graph(expression)
    for i in range(1, len(G.nodes()) + 1):
        if i == len(G.nodes()):
            G.nodes[i]['question_node'] = 1
        else:
            G.nodes[i]['question_node'] = 0
    return G


def _get_graph(
        expression: List) -> nx.MultiGraph:  # The id of question node is always the same as the size of the graph
    global id2name_dict
    if isinstance(expression, str):
        G = nx.MultiDiGraph()
        if get_symbol_type(expression) == 1:
            if id2name_dict is not None:
                if expression in id2name_dict:
                    name = id2name_dict[expression]
                else:
                    name = "null"
                G.add_node(1, mid=expression, node_type='entity', function='none',
                           friendly_name=name, label="{} ({})".format(expression, name))
            else:
                G.add_node(1, mid=expression, node_type='entity', function='none', label=expression)
        elif get_symbol_type(expression) == 2:
            G.add_node(1, mid=expression, node_type='literal')
        elif get_symbol_type(expression) == 3:
            G.add_node(1, mid=expression, node_type='class', function='none')
        elif get_symbol_type(expression) == 4:  # relation or attribute
            if expression in relation_dr.keys():
                domain, rang = relation_dr[expression]
            else:
                domain, rang = 'a', 'b'
            G.add_node(1, mid=rang, node_type='class',
                       function='none')  # if it's an attribute, the type will be changed to literal in arg
            G.add_node(2, mid=domain, node_type='class', function='none')
            G.add_edge(2, 1, label=expression)

            if REVERSE:
                if expression in reverse_properties:
                    G.add_edge(1, 2, label=reverse_properties[expression])

        return G

    if expression[0] == 'R':
        G = _get_graph(expression[1])
        size = len(G.nodes())
        mapping = {}
        for n in G.nodes():
            mapping[n] = size - n + 1
        G = nx.relabel_nodes(G, mapping)
        return G

    elif expression[0] in ['JOIN', 'le', 'ge', 'lt', 'gt']:
        G1 = _get_graph(expression=expression[1])
        G2 = _get_graph(expression=expression[2])
        size = len(G2.nodes())
        qn_id = size
        if G1.nodes[1]['node_type'] == G2.nodes[qn_id]['node_type'] == 'class':
            if G2.nodes[qn_id]['mid'] in upper_types[G1.nodes[1]['mid']]:
                G2.nodes[qn_id]['mid'] = G1.nodes[1]['mid']
            # G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G = nx.compose(G1, G2)

        if expression[0] != 'JOIN':
            G.nodes[1]['function'] = function_map[expression[0]]

        return G

    elif expression[0] == 'AND':
        G1 = _get_graph(expression[1])
        G2 = _get_graph(expression[2])

        size1 = len(G1.nodes())
        size2 = len(G2.nodes())
        if G1.nodes[size1]['node_type'] == G2.nodes[size2]['node_type'] == 'class':
            # if G2.nodes[size2]['id'] in upper_types[G1.nodes[size1]['id']]:
            G2.nodes[size2]['mid'] = G1.nodes[size1]['mid']
            # IIRC, in nx.compose, for the same node, its information can be overwritten by its info in the second graph
            # So here for the AND function we force it to choose the type explicitly provided in the logical form
        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size2 - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
        G = nx.compose(G1, G2)

        return G

    elif expression[0] == 'COUNT':
        G = _get_graph(expression[1])
        size = len(G.nodes())
        G.nodes[size]['function'] = 'count'

        return G

    elif expression[0].__contains__('ARG'):
        G1 = _get_graph(expression[1])
        size1 = len(G1.nodes())

        G2 = _get_graph(expression[2])
        size2 = len(G2.nodes())

        # G2.nodes[1]['class'] = G2.nodes[1]['id']   # not sure whether this is needed for sparql
        G2.nodes[1]['mid'] = 0
        G2.nodes[1]['node_type'] = 'literal'
        G2.nodes[1]['function'] = expression[0].lower()
        if G1.nodes[size1]['node_type'] == G2.nodes[size2]['node_type'] == 'class':
            # if G2.nodes[size2]['id'] in upper_types[G1.nodes[size1]['id']]:
            G2.nodes[size2]['mid'] = G1.nodes[size1]['mid']

        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size2 - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
        G = nx.compose(G1, G2)

        return G

    elif expression[0] == 'TC':
        G = _get_graph(expression[1])
        size = len(G.nodes())
        G.nodes[size]['tc'] = (expression[2], expression[3])

        return G


def get_symbol_type(symbol: str) -> int:
    if symbol.__contains__('^^'):
        return 2
    elif symbol in types:
        return 3
    elif symbol in relations or len(list(symbol.split('.'))) > 2:
        return 4
    elif symbol:
        return 1


def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]


def binary_nesting(function: str, elements: List[str], types_along_path=None) -> str:
    if len(elements) < 2:
        print("error: binary function should have 2 parameters!")
    if not types_along_path:
        if len(elements) == 2:
            return '(' + function + ' ' + elements[0] + ' ' + elements[1] + ')'
        else:
            return '(' + function + ' ' + elements[0] + ' ' + binary_nesting(function, elements[1:]) + ')'
    else:
        if len(elements) == 2:
            return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' + elements[1] + ')'
        else:
            return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' \
                   + binary_nesting(function, elements[1:], types_along_path[1:]) + ')'


def set_visited(G, s, e, relation):
    end_num = get_end_num(G, s)
    for i in range(0, end_num[e]):
        if G.edges[s, e, i]['relation'] == relation:
            G.edges[s, e, i]['visited'] = True


def get_end_num(G, s):
    end_num = defaultdict(lambda: 0)
    for edge in list(G.edges(s)):  # for directed graph G.edges is the same as G.out_edges, not including G.in_edges
        end_num[list(edge)[1]] += 1
    return end_num


def none_function(G, start, arg_node=None, type_constraint=False):
    if arg_node is not None:
        arg = G.nodes[arg_node]['function']
        path = list(nx.all_simple_paths(G, start, arg_node))
        assert len(path) == 1
        arg_clause = []
        for i in range(0, len(path[0]) - 1):
            edge = G.edges[path[0][i], path[0][i + 1], 0]
            if edge['reverse']:
                relation = '(R ' + edge['relation'] + ')'
            else:
                relation = edge['relation']
            arg_clause.append(relation)

        # Deleting edges until the first node with out degree > 2 is meet
        # (conceptually it should be 1, but remember that add edges is both directions)
        while i >= 0:
            flag = False
            if G.out_degree[path[0][i]] > 2:
                flag = True
            G.remove_edge(path[0][i], path[0][i + 1], 0)
            i -= 1
            if flag:
                break

        if len(arg_clause) > 1:
            arg_clause = binary_nesting(function='JOIN', elements=arg_clause)
            # arg_clause = ' '.join(arg_clause)
        else:
            arg_clause = arg_clause[0]

        return '(' + arg.upper() + ' ' + none_function(G, start) + ' ' + arg_clause + ')'

    if G.nodes[start]['type'] != 'class':
        return G.nodes[start]['id']

    end_num = get_end_num(G, start)
    clauses = []

    if G.nodes[start]['question'] and type_constraint:
        clauses.append(G.nodes[start]['id'])
    for key in end_num.keys():
        for i in range(0, end_num[key]):
            if not G.edges[start, key, i]['visited']:
                relation = G.edges[start, key, i]['relation']
                G.edges[start, key, i]['visited'] = True
                set_visited(G, key, start, relation)
                if G.edges[start, key, i]['reverse']:
                    relation = '(R ' + relation + ')'
                if G.nodes[key]['function'].__contains__('<') or G.nodes[key]['function'].__contains__('>'):
                    if G.nodes[key]['function'] == '>':
                        clauses.append('(gt ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '>=':
                        clauses.append('(ge ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '<':
                        clauses.append('(lt ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '<=':
                        clauses.append('(le ' + relation + ' ' + none_function(G, key) + ')')
                else:
                    # try:
                    clauses.append('(JOIN ' + relation + ' ' + str(none_function(G, key)) + ')')
                    # except Exception:
                    #     print(relation, none_function(G, key), json_graph.node_link_data(G))

    if len(clauses) == 0:
        return G.nodes[start]['id']

    if len(clauses) == 1:
        return clauses[0]
    else:
        return binary_nesting(function='AND', elements=clauses)


def count_function(G, start):
    return '(COUNT ' + none_function(G, start) + ')'


def get_lisp_from_graph_query_new(graph_query):
    G = nx.MultiDiGraph()
    aggregation = 'none'
    arg_node = None
    for node in graph_query['nodes']:
        G.add_node(node['id'], id=node['mid'], type=node['node_type'], question=node['question_node'],
                   function=node['function'])
        if node['question_node'] == 1:
            qid = node['id']
        if node['function'] != 'none':
            aggregation = node['function']
            if node['function'].__contains__('arg'):
                arg_node = node['id']
    for edge in graph_query['links']:
        G.add_edge(edge['source'], edge['target'], relation=edge['label'], reverse=False, visited=False)
        G.add_edge(edge['target'], edge['source'], relation=edge['label'], reverse=True, visited=False)
    if 'count' == aggregation:
        return count_function(G, qid)
    else:
        # print(none_function(G, qid))
        return none_function(G, qid, arg_node=arg_node)


def lisp_to_sparql(lisp_program: str):
    def _linearize_lisp_expression(expression: list, sub_formula_id):
        sub_formulas = []
        for i, e in enumerate(expression):
            if isinstance(e, list) and e[0] != 'R':
                sub_formulas.extend(_linearize_lisp_expression(e, sub_formula_id))
                expression[i] = '#' + str(sub_formula_id[0] - 1)

        sub_formulas.append(expression)
        sub_formula_id[0] += 1
        return sub_formulas

    clauses = []
    order_clauses = []
    entities = set()  # collect entites for filtering
    # identical_variables = {}   # key should be smaller than value, we will use small variable to replace large variable
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    superlative = False
    if expression[0] in ['ARGMAX', 'ARGMIN']:
        superlative = True
        # remove all joins in relation chain of an arg function. In another word, we will not use arg function as
        # binary function here, instead, the arity depends on the number of relations in the second argument in the
        # original function
        if isinstance(expression[2], list):
            def retrieve_relations(exp: list):
                rtn = []
                for element in exp:
                    if element == 'JOIN':
                        continue
                    elif isinstance(element, str):
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'R':
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'JOIN':
                        rtn.extend(retrieve_relations(element))
                return rtn

            relations = retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    count = False

    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + subp[2][1:] + " ns:" + subp[1][1] + " ?x" + i + " .")
                else:  # literal   (actually I think literal can only be object)
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime', 'double']:
                            subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                            # subp[2] = subp[2].split("^^")[0] + '-08:00^^' + subp[2].split("^^")[1]
                        else:
                            subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
                    clauses.append(subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
            else:
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + i + " ns:" + subp[1] + " ?x" + subp[2][1:] + " .")
                else:  # literal
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime', 'double']:
                            subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                        else:
                            subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
                    clauses.append("?x" + i + " ns:" + subp[1] + " " + subp[2] + " .")
        elif subp[0] == 'AND':
            var1 = int(subp[2][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            # identical_variables[var1] = int(i)
            if subp[1][0] == "#":
                var2 = int(subp[1][1:])
                root2 = get_root(var2)
                # identical_variables[var2] = int(i)
                if root1 > root2:
                    # identical_variables[var2] = var1
                    identical_variables_r[root1] = root2
                else:
                    # identical_variables[var1] = var2
                    identical_variables_r[root2] = root1
            else:  # 2nd argument is a class
                clauses.append("?x" + i + " ns:type.object.type ns:" + subp[1] + " .")
        elif subp[0] in ['le', 'lt', 'ge', 'gt']:  # the 2nd can only be numerical value
            clauses.append("?x" + i + " ns:" + subp[1] + " ?y" + i + " .")
            if subp[0] == 'le':
                op = "<="
            elif subp[0] == 'lt':
                op = "<"
            elif subp[0] == 'ge':
                op = ">="
            else:
                op = ">"
            if subp[2].__contains__('^^'):
                data_type = subp[2].split("^^")[1].split("#")[1]
                if data_type not in ['integer', 'float', 'dateTime', 'double']:
                    subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                else:
                    subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
            clauses.append(f"FILTER (?y{i} {op} {subp[2]})")
        elif subp[0] == 'TC':
            var = int(subp[1][1:])
            # identical_variables[var] = int(i)
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti

            year = subp[3]
            if year == 'NOW':
                from_para = '"2015-08-10"^^xsd:dateTime'
                to_para = '"2015-08-10"^^xsd:dateTime'
            else:
                from_para = f'"{year}-12-31"^^xsd:dateTime'
                to_para = f'"{year}-01-01"^^xsd:dateTime'

            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2]} ?sk0}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{subp[2]} ?sk1 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
            if subp[2][-4:] == "from":
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk3 . ')
            else:  # from_date -> to_date
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk3 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            superlative = True
            if subp[1][0] == '#':
                var = int(subp[1][1:])
                rooti = get_root(int(i))
                root_var = get_root(var)
                # identical_variables[var] = int(i)
                if rooti > root_var:
                    identical_variables_r[rooti] = root_var
                else:
                    identical_variables_r[root_var] = rooti
            else:  # arg1 is class
                clauses.append(f'?x{i} ns:type.object.type ns:{subp[1]} .')

            if len(subp) == 3:
                clauses.append(f'?x{i} ns:{subp[2]} ?sk0 .')
            elif len(subp) > 3:
                for j, relation in enumerate(subp[2:-1]):
                    if j == 0:
                        var0 = f'x{i}'
                    else:
                        var0 = f'c{j - 1}'
                    var1 = f'c{j}'
                    if isinstance(relation, list) and relation[0] == 'R':
                        clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                    else:
                        clauses.append(f'?{var0} ns:{relation} ?{var1} .')

                clauses.append(f'?c{j} ns:{subp[-1]} ?sk0 .')

            if subp[0] == 'ARGMIN':
                order_clauses.append("ORDER BY ?sk0")
            elif subp[0] == 'ARGMAX':
                order_clauses.append("ORDER BY DESC(?sk0)")
            order_clauses.append("LIMIT 1")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')

    if superlative:
        arg_clauses = clauses[:]

    for entity in entities:
        clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        clauses.insert(0, f"SELECT COUNT DISTINCT ?x")
    elif superlative:
        clauses.insert(0, "{SELECT ?sk0")
        clauses = arg_clauses + clauses
        clauses.insert(0, "WHERE {")
        clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    clauses.append('}')
    clauses.extend(order_clauses)
    if superlative:
        clauses.append('}')
        clauses.append('}')

    # for clause in clauses:
    #     print(clause)

    return '\n'.join(clauses)


def process_ontology(fb_roles_file, fb_types_file, reverse_properties_file):
    reverse_properties = {}
    with open(reverse_properties_file, 'r') as f:
        for line in f:
            reverse_properties[line.split('\t')[0]] = line.split('\t')[1].replace('\n', '')

    with open(fb_roles_file, 'r') as f:
        content = f.readlines()

    relation_dr = {}
    relations = set()
    for line in content:
        fields = line.split()
        relation_dr[fields[1]] = (fields[0], fields[2])
        relations.add(fields[1])

    with open(fb_types_file, 'r') as f:
        content = f.readlines()

    upper_types = defaultdict(lambda: set())

    types = set()
    for line in content:
        fields = line.split()
        upper_types[fields[0]].add(fields[2])
        types.add(fields[0])
        types.add(fields[2])

    return reverse_properties, relation_dr, relations, upper_types, types


class SemanticMatcher:
    def __init__(self, reverse_properties, relation_dr, relations, upper_types, types):
        self.reverse_properties = reverse_properties
        self.relation_dr = relation_dr
        self.relations = relations
        self.upper_types = upper_types
        self.types = types

    def same_logical_form(self, form1, form2):
        if form1.__contains__("@@UNKNOWN@@") or form2.__contains__("@@UNKNOWN@@"):
            return False
        try:
            G1 = self.logical_form_to_graph(lisp_to_nested_expression(form1))
        except Exception:
            return False
        try:
            G2 = self.logical_form_to_graph(lisp_to_nested_expression(form2))
        except Exception:
            return False

        def node_match(n1, n2):
            if n1['id'] == n2['id'] and n1['type'] == n2['type']:
                func1 = n1.pop('function', 'none')
                func2 = n2.pop('function', 'none')
                tc1 = n1.pop('tc', 'none')
                tc2 = n2.pop('tc', 'none')

                if func1 == func2 and tc1 == tc2:
                    return True
                else:
                    return False
                # if 'function' in n1 and 'function' in n2 and n1['function'] == n2['function']:
                #     return True
                # elif 'function' not in n1 and 'function' not in n2:
                #     return True
                # else:
                #     return False
            else:
                return False

        def multi_edge_match(e1, e2):
            if len(e1) != len(e2):
                return False
            values1 = []
            values2 = []
            for v in e1.values():
                values1.append(v['relation'])
            for v in e2.values():
                values2.append(v['relation'])
            return sorted(values1) == sorted(values2)

        return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=multi_edge_match)

    def get_symbol_type(self, symbol: str) -> int:
        if symbol.__contains__('^^'):  # literals are expected to be appended with data types
            return 2
        elif symbol in self.types:
            return 3
        elif symbol in self.relations:
            return 4
        else:
            return 1

    def logical_form_to_graph(self, expression: List) -> nx.MultiGraph:
        # TODO: merge two entity node with same id. But there is no such need for
        # the second version of graphquestions
        G = self._get_graph(expression)
        G.nodes[len(G.nodes())]['question_node'] = 1
        return G

    def _get_graph(self,
                   expression: List) -> nx.MultiGraph:  # The id of question node is always the same as the size of the graph
        if isinstance(expression, str):
            G = nx.MultiDiGraph()
            if self.get_symbol_type(expression) == 1:
                G.add_node(1, id=expression, type='entity')
            elif self.get_symbol_type(expression) == 2:
                G.add_node(1, id=expression, type='literal')
            elif self.get_symbol_type(expression) == 3:
                G.add_node(1, id=expression, type='class')
                # G.add_node(1, id="common.topic", type='class')
            elif self.get_symbol_type(expression) == 4:  # relation or attribute
                domain, rang = self.relation_dr[expression]
                G.add_node(1, id=rang, type='class')  # if it's an attribute, the type will be changed to literal in arg
                G.add_node(2, id=domain, type='class')
                G.add_edge(2, 1, relation=expression)

                if expression in self.reverse_properties:  # take care of reverse properties
                    G.add_edge(1, 2, relation=self.reverse_properties[expression])

            return G

        if expression[0] == 'R':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            mapping = {}
            for n in G.nodes():
                mapping[n] = size - n + 1
            G = nx.relabel_nodes(G, mapping)
            return G

        elif expression[0] in ['JOIN', 'le', 'ge', 'lt', 'gt']:
            G1 = self._get_graph(expression=expression[1])
            G2 = self._get_graph(expression=expression[2])
            size = len(G2.nodes())
            qn_id = size
            if G1.nodes[1]['type'] == G2.nodes[qn_id]['type'] == 'class':
                if G2.nodes[qn_id]['id'] in self.upper_types[G1.nodes[1]['id']]:
                    G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
                # G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G = nx.compose(G1, G2)

            if expression[0] != 'JOIN':
                G.nodes[1]['function'] = function_map[expression[0]]

            return G

        elif expression[0] == 'AND':
            G1 = self._get_graph(expression[1])
            G2 = self._get_graph(expression[2])

            size1 = len(G1.nodes())
            size2 = len(G2.nodes())
            if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
                G2.nodes[size2]['id'] = G1.nodes[size1]['id']
                # IIRC, in nx.compose, for the same node, its information can be overwritten by its info in the second graph
                # So here for the AND function we force it to choose the type explicitly provided in the logical form
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2 - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
            G = nx.compose(G1, G2)

            return G

        elif expression[0] == 'COUNT':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            G.nodes[size]['function'] = 'count'

            return G

        elif expression[0].__contains__('ARG'):
            G1 = self._get_graph(expression[1])
            size1 = len(G1.nodes())
            G2 = self._get_graph(expression[2])
            size2 = len(G2.nodes())
            # G2.nodes[1]['class'] = G2.nodes[1]['id']   # not sure whether this is needed for sparql
            G2.nodes[1]['id'] = 0
            G2.nodes[1]['type'] = 'literal'
            G2.nodes[1]['function'] = expression[0].lower()
            if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
                G2.nodes[size2]['id'] = G1.nodes[size1]['id']

            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2 - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
            G = nx.compose(G1, G2)

            return G

        elif expression[0] == 'TC':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            G.nodes[size]['tc'] = (expression[2], expression[3])

            return G


MACRO_TEMPLATES = [('american_football.football_historical_coach_position', 'from', 'to'),
                   ('architecture.ownership', 'start_date', 'end_date'),
                   ('award.award_honor', 'year', 'year'), ('business.employment_tenure', 'from', 'to'),
                   ('business.sponsorship', 'from', 'to'),
                   ('celebrities.romantic_relationship', 'start_date', 'end_date'),
                   ('chemistry.chemical_element', 'discovery_date', 'discovery_date'),
                   ('film.film', 'initial_release_date', 'initial_release_date'),
                   ('government.government_position_held', 'from', 'to'),
                   ('law.invention', 'date_of_invention', 'date_of_invention'),
                   ('law.judicial_tenure', 'from_date', 'to_date'),
                   ('organization.organization_relationship', 'to', 'from'), ('people.marriage', 'from', 'to'),
                   ('people.place_lived', 'end_date', 'start_date'), ('sports.sports_team_coach_tenure', 'from', 'to'),
                   ('sports.sports_team_roster', 'from', 'to'), ('sports.team_venue_relationship', 'from', 'to'),
                   ('time.event', 'start_date', 'end_date'), ('tv.regular_tv_appearance', 'from', 'to'),
                   ('tv.tv_network_duration', 'from', 'to')]
MACRO_TEMPLATES = dict([(x[0], (x[1], x[2])) for x in MACRO_TEMPLATES])

LEVEL_MACRO_CLAUSES = '''
FILTER(NOT EXISTS {{{var} ns:{relation}.{start_suffix} ?sk6}} || 
EXISTS {{{var} ns:{relation}.{start_suffix} ?sk7 . 
FILTER(xsd:datetime(?sk7) <= "{year}-12-31"^^xsd:dateTime) }})
FILTER(NOT EXISTS {{{var} ns:{relation}.{end_suffix} ?sk8}} || 
EXISTS {{{var} ns:{relation}.{end_suffix} ?sk9 . 
FILTER(xsd:datetime(?sk9) >= "{year}-01-01"^^xsd:dateTime) }})
'''


def get_time_macro_clause(x):
    def _get_time_macro_clause(node):
        # print('NODE', node.construction, node.logical_form())
        if (node.construction == 'AND' and
                node.fields[0].construction == 'JOIN' and
                node.fields[0].fields[0].construction == 'SCHEMA' and
                'time_macro' in node.fields[0].fields[0].val):
            return node.fields[0]
        else:
            for field in node.fields:
                ret_val = _get_time_macro_clause(field)
                if ret_val is not None:
                    return ret_val
            return None

    assert '.time_macro' in x

    ast = parse_s_expr(x)
    macro_node = _get_time_macro_clause(ast)
    assert macro_node is not None

    relation = macro_node.fields[0].val
    year = macro_node.fields[1].val[:4]
    assert relation.endswith('.time_macro')
    relation = '.'.join(relation.split('.')[:2])
    start_end_suffix = MACRO_TEMPLATES.get(relation, ('from', 'to'))
    # if macro_node
    if macro_node.level == 1:
        var = '?x'
    else:
        var = '?x0'

    additional_clause = LEVEL_MACRO_CLAUSES.format(var=var, relation=relation, start_suffix=start_end_suffix[0],
                                                   end_suffix=start_end_suffix[1], year=year)
    return additional_clause


def get_approx_s_expr(x):
    def approx_time_macro_ast(node):
        # print('NODE', node.construction, node.logical_form())
        if (node.construction == 'AND' and
                node.fields[0].construction == 'JOIN' and
                node.fields[0].fields[0].construction == 'SCHEMA' and
                'time_macro' in node.fields[0].fields[0].val):
            return node.fields[1]
        else:
            new_fileds = [approx_time_macro_ast(x) for x in node.fields]
            node.fields = new_fileds
            return node

    if not ('time_macro' in x):
        return x

    ast = parse_s_expr(x)
    approx_ast = approx_time_macro_ast(ast)
    approx_x = approx_ast.compact_logical_form()
    return approx_x


def s_expression_to_sparql(s_expression: str):
    if 'time_macro' in s_expression:
        approx_expr = get_approx_s_expr(s_expression)
        additional_clause = get_time_macro_clause(s_expression)
        approx_sparql = lisp_to_sparql(approx_expr)
        approx_sparql_end = approx_sparql.rfind('}')
        sparql = approx_sparql[:approx_sparql_end] + additional_clause + approx_sparql[approx_sparql_end:]
    else:
        sparql = lisp_to_sparql(s_expression)
    return sparql


def clear_arg(query_expression: str):
    query_expression = "(AND" + query_expression[7:]
    query_expression = query_expression.replace("?r", "(JOIN ?r ?z)")
    return query_expression
