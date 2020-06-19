import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from data_pre_process.my_ast import read_pickle, traverse_tree_to_generate_matrix
from train.evaluation import load_json
from model.utils import pad_seq, subsequent_mask, make_std_mask


class TreeDataSet(Dataset):
    def __init__(self,
                 file_name,
                 ast_path,
                 ast2id,
                 nl2id,
                 max_ast_size,
                 max_simple_name_size,
                 k,
                 max_comment_size,
                 use_code):
        """
        :param file_name: 数据集名称
        :param ast_path: AST存放路径
        :param max_ast_size: 最大AST节点数
        :param k: 最大相对位置
        :param max_comment_size: 最大评论长度
        """
        super(TreeDataSet, self).__init__()
        print('loading data...')
        self.data_set = load_json(file_name)[:10]
        print('loading data finished...')

        self.max_ast_size = max_ast_size
        self.k = k
        self.max_comment_size = max_comment_size
        self.ast_path = ast_path
        self.ast2id = ast2id
        self.nl2id = nl2id
        self.max_simple_name_size = max_simple_name_size

        self.use_code = use_code

        self.len = len(self.data_set)

    def __getitem__(self, index):
        data = self.data_set[index]
        ast_num = data['ast_num']
        nl = data['nl']

        ast = read_pickle(self.ast_path + ast_num)
        seq, rel_par, rel_bro, rel_semantic, semantic_convert_matrix, semantic_mask = traverse_tree_to_generate_matrix(ast, self.max_ast_size, self.k, self.max_simple_name_size)

        seq_id = [self.ast2id[x] if x in self.ast2id else self.ast2id['<UNK>'] for x in seq]
        nl_id = [self.nl2id[x] if x in self.nl2id else self.nl2id['<UNK>'] for x in nl]

        """to tensor"""
        seq_tensor = torch.LongTensor(seq_id)
        nl_tensor = torch.LongTensor(pad_seq(nl_id, self.max_comment_size).long())

        return seq_tensor, nl_tensor, rel_par, rel_bro, rel_semantic, semantic_convert_matrix, semantic_mask

    def __len__(self):
        return self.len

    @staticmethod
    def make_std_mask(comment, pad):
        comment_mask = (comment != pad).unsqueeze(-2)
        tgt_mask = comment_mask & Variable(
            subsequent_mask(comment.size(-1)).type_as(comment_mask.data))
        return tgt_mask


def collate_fn(inputs):
    codes = []
    nls = []
    rel_pars = []
    rel_bros = []
    rel_semantics = []
    semantic_converts = []
    semantic_masks = []

    for i in range(len(inputs)):
        code, nl, rel_par, rel_bro, rel_semantic, semantic_convert, semantic_mask = inputs[i]

        codes.append(code)
        nls.append(nl)
        rel_pars.append(rel_par)
        rel_bros.append(rel_bro)
        rel_semantics.append(rel_semantic)
        semantic_converts.append(semantic_convert)
        semantic_masks.append(semantic_mask)

    batch_code = torch.stack(codes, dim=0)
    batch_nl = torch.stack(nls, dim=0)

    batch_comments = batch_nl[:, :-1]
    batch_predicts = batch_nl[:, 1:]

    comment_mask = make_std_mask(batch_comments, 0)
    comment_mask = comment_mask.unsqueeze(1) == 0

    re_par_ids = torch.stack(rel_pars, dim=0)
    re_bro_ids = torch.stack(rel_bros, dim=0)
    rel_semantic_ids = torch.stack(rel_semantics, dim=0)
    semantic_converts = torch.stack(semantic_converts, dim=0)
    semantic_masks = torch.stack(semantic_masks, dim=0)

    return (batch_code, re_par_ids, re_bro_ids, rel_semantic_ids, semantic_converts, semantic_masks, batch_comments, comment_mask), batch_predicts
