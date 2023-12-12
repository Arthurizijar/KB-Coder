import argparse
import ruamel.yaml as yaml
from dataset import DataSet
from selector import Selector
from relation_linker import RelationLinker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config')
    parser.add_argument('--link_config', default='./configs/Link.yaml')
    args = parser.parse_args()
    data_config = yaml.load(open(args.data_config, 'r'), Loader=yaml.Loader)
    link_config = yaml.load(open(args.link_config, 'r'), Loader=yaml.Loader)
    train_set = DataSet(data_config, type='train')
    test_set = DataSet(data_config, type='test')
    selector = Selector(data_config)
    relation_linker = RelationLinker(link_config)
    json_query = selector.queries_gen(data_config, train_set, test_set, relation_linker, export_answer=False)
