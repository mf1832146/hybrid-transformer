import argparse

from data_pre_process.my_ast import read_pickle
from train.solver import Solver


def parse():
    parser = argparse.ArgumentParser(description='tree transformer')
    parser.add_argument('-model_dir', default='train_model', help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-model', default='hybrid-transformer', help='[hybrid-transformer]')
    parser.add_argument('-num_step', type=int, default=250)
    parser.add_argument('-num_layers', type=int, default=2, help='layer num')
    parser.add_argument('-model_dim', type=int, default=384)
    parser.add_argument('-num_heads', type=int, default=6)
    parser.add_argument('-ffn_dim', type=int, default=1536)

    parser.add_argument('-data_dir', default='../data_set')
    parser.add_argument('-code_max_len', type=int, default=200, help='max length of code')
    parser.add_argument('-comment_max_len', type=int, default=30, help='comment max length')
    parser.add_argument('-relative_pos', type=bool, default=True, help='use relative position')
    parser.add_argument('-k', type=int, default=5, help='relative window size')
    parser.add_argument('-max_simple_name_len', type=int, default=30, help='max simple name length')

    parser.add_argument('-dropout', type=float, default=0.5)

    parser.add_argument('-load', action='store_true', help='load pretrained model')
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-visual', action='store_true')
    parser.add_argument('-gold_test', action='store_true')

    parser.add_argument('-load_epoch', type=str, default='0')

    parser.add_argument('-log_dir', default='train_log/')

    parser.add_argument('-g', type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    i2code = read_pickle(args.data_dir + '/code_i2w.pkl')
    i2nl = read_pickle(args.data_dir + '/nl_i2w.pkl')
    i2ast = read_pickle(args.data_dir + '/ast_i2w.pkl')

    ast2id = {v: k for k, v in i2ast.items()}
    code2id = {v: k for k, v in i2code.items()}
    nl2id = {v: k for k, v in i2nl.items()}

    solver = Solver(args, ast2id, code2id, nl2id, i2nl)

    if args.train:
        solver.train()
    elif args.test:
        solver.test(load_epoch=args.load_epoch)
    elif args.visual:
        solver.visualize(load_epoch=args.load_epoch)
    elif args.gold_test:
        solver.gold_test(load_epoch=args.load_epoch)

