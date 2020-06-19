import os
import json
from collections import Counter
from glob import glob

from tqdm import tqdm
from data_pre_process.my_ast import parse, traverse_label, get_method_name, get_bracket, get_identifier, get_values, \
    traverse, Node, traverse_simple_name, sub_tree
import re
import wordninja
import pickle
import nltk
lemmer = nltk.stem.WordNetLemmatizer()


def process(data_dir='../data', output_data_dir='../data_set', max_size=200, max_simple_len=30, vocab_size=30000):
    """../data_dir是存放原始AST的目录，../data_set是存放处理后AST的目录"""
    dirs = [
        output_data_dir,
        output_data_dir + '/tree',
        output_data_dir + '/tree/train',
        output_data_dir + '/tree/valid',
        output_data_dir + '/tree/test'
    ]

    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    for path in [data_dir + "/" + s for s in ["train", "valid", "test"]]:
        set_name = path.split("/")[-1]
        outputs = parse_dir(path, output_data_dir, max_size, max_simple_len, vocab_size)
        with open(output_data_dir + '/' + set_name + '.json', 'w') as f:
            json.dump(outputs, f)


def parse_dir(original_data_dir, output_data_dir, max_size=200, max_simple_len=30, vocab_size=30000):
    files = sorted(glob(original_data_dir + "/*"))
    set_name = original_data_dir.split("/")[-1]

    # nls用来生成注释词表， codes用来生成源代码词表（本模型未使用）,asts用来生成AST词表
    nls = []
    codes = []
    asts = []

    outputs = []
    skip = 0

    for file in tqdm(files, "parsing {}".format(original_data_dir)):
        tree, code, nl = parse(file)
        if tree is None:
            skip += 1
            continue
        # 跳过以/*开头的注释
        if is_invalid_com(nl):
            skip += 1
            continue
        # 跳过节点数大于1000的树，或者以test,set开头的方
        if is_invalid_tree(tree):
            skip += 1
            continue

        """重新生成树, 对变量名，value节点进行转化"""
        rebuild_tree(tree)
        """取出最大子树"""
        sub_tree(tree, max_size=max_size)
        """"""
        # 多余一行的注释只保留第一行
        nl = clean_nl(nl)
        # 分词
        seq = tokenize(nl)
        # 跳过小于4个词的注释
        if is_invalid_seq(seq):
            skip += 1
            continue
        # 将变量名, 方法名, value转化为<SimpleName>_0, <SimpleName>_1, ...
        seq = replace_simple_name(seq, tree, max_simple_len=max_simple_len)

        print(1)
        number = int(file.split("\/")[-1])  # AST对应存储的文件名
        code_token = tokenize(code)[1:-1]  # 对源代码进行分词，本模型未使用，为了其他基线模型

        ast_num = str(number)

        nls.append(seq)
        codes.append(code_token)
        asts.append(traverse_label(tree))

        outputs.append({'ast_num': ast_num,
                        'code': code_token,
                        'nl': seq})

        """将AST树存储到tree目录下"""
        with open(output_data_dir + "/tree/" + set_name + "/" + ast_num, "wb", 1) as f:
            pickle.dump(tree, f)

    print("{} files skipped".format(skip))
    # 生成训练词表
    if set_name == 'train':
        # 1. 生成AST词表
        ast_counter = Counter([l for s in asts for l in s])
        ast_vocab = {i: w for i, w in enumerate(
            ["<PAD>", "<UNK>"] + sorted([x[0] for x in ast_counter.most_common(vocab_size)]))}
        # 2. 生成注释词表
        nl_tokens = []
        for l in nls:
            for x in l:
                if not x.startswith('<SimpleName>_'):
                    nl_tokens.append(x)

        nl_counter = Counter(nl_tokens)
        nl_tokens = sorted(x[0] for x in nl_counter.most_common(vocab_size))

        nl_tokens += ['<SimpleName>_' + str(i) for i in range(max_simple_len)]
        nl_vocab = {i: w for i, w in enumerate(
            ["<PAD>", "<UNK>"] + nl_tokens)}
        # 3. 生成源代码词表（本模型未使用）
        code_counter = Counter([x for l in codes for x in l])
        code_vocab = {i: w for i, w in enumerate(
            ["<PAD>", "<UNK>"] + sorted([x[0] for x in code_counter.most_common(vocab_size)]))}

        print('ast_vocab', list(ast_vocab.items()))
        print('nl_vocab', list(nl_vocab.items()))

        # 保存
        pickle.dump(ast_vocab, open(output_data_dir + "/ast_i2w.pkl", "wb"))
        pickle.dump(nl_vocab, open(output_data_dir + "/nl_i2w.pkl", "wb"))
        pickle.dump(code_vocab, open(output_data_dir + "/code_i2w.pkl", "wb"))

    return outputs


def replace_simple_name(seq, root, max_simple_len=30):
    simple_names = traverse_simple_name(root)[: max_simple_len]
    new_seq = []
    for s in seq:
        if s in simple_names:
            new_seq.append('<SimpleName>_' + str(simple_names.index(s)))
        else:
            new_seq.append(s)
    return new_seq


def rebuild_tree(root):
    for node in traverse(root):
        if "=" not in node.label and "(SimpleName)" in node.label:
            id_ = get_identifier(node.children[0].label)
            if id_ is None:
                raise Exception("ERROR!")

            node.is_simple_name = True
            # 将变量名保存到自身
            node.simple_name = id_

            # 将simpleName的子节点拆分为多个单词节点
            words = deal_with_simple_name(id_)
            node.children = [Node(label=w, parent=node, is_leaf_node=True, children=[]) for w in words]

        elif node.label[:6] == "value=":
            value_ = get_values(node.label)
            node.is_simple_name = True
            node.simple_name = value_
            if isnum(value_):
                node.label = 'Value_<NUM>'
                node.children = [Node(label=value_, parent=node, is_leaf_node=True, children=[])]
            else:
                """字符串"""
                words = deal_with_simple_name(value_)
                node.children = [Node(label=w, parent=node, is_leaf_node=True, children=[]) for w in words]
                node.label = 'Value_<STR>'
    return root


def clean_nl(s):
    if s[-1] == ".":
        s = s[:-1]
    s = s.split(". ")[0]
    s = re.sub("[<].+?[>]", "", s)
    s = re.sub("[\[\]\%]", "", s)
    s = s[0:1].lower() + s[1:]
    return s


def is_invalid_com(s):
    return s[:2] == "/*" and len(s) > 1


def is_invalid_tree(root):
    labels = traverse_label(root)
    if root.label == 'root (ConstructorDeclaration)':
        return True
    if len(labels) >= 1000:
        return True
    method_name = get_method_name(root)
    for word in ["test", "Test", "set", "Set", "get", "Get"]:
        if method_name[:len(word)] == word:
            return True
    return False


def is_invalid_seq(s):
    return len(s) < 4


def deal_with_simple_name(s):
    """This function is used to deal with simpleName includes 变量名和方法名
        by using wordninja
    """
    return wordninja.split(s.lower())


def tokenize(s):
    tokens = nltk.word_tokenize(s)
    # 规范化词
    standard_tokens = ['<s>'] + [lemmer.lemmatize(token) for token in tokens] + ['</s>']
    return standard_tokens


def isnum(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


if __name__ == '__main__':
    process(data_dir='../data', output_data_dir='../data_set', max_size=200, max_simple_len=30, vocab_size=200)
