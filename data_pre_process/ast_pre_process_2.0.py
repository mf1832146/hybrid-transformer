import os
import json
from collections import Counter
from glob import glob

from tqdm import tqdm
from data_pre_process.my_ast import parse, traverse_label, get_method_name, get_bracket, get_identifier, get_values, \
    traverse, Node, traverse_simple_name, sub_tree, read_pickle
import re
import wordninja
import pickle
from pytorch_pretrained_bert import BasicTokenizer

tokenizer = BasicTokenizer(do_lower_case=True)


def process(data_dir='../data', output_data_dir='../data_set', max_size=100, max_simple_len=30, vocab_size=30000):
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


def parse_dir(original_data_dir, output_data_dir, max_size=100, max_simple_len=30, vocab_size=30000, min_count=10):
    """min_count是进入词表的最小值"""
    files = sorted(glob(original_data_dir + "/*"))
    set_name = original_data_dir.split("/")[-1]

    # nls用来生成注释词表， codes用来生成源代码词表（本模型未使用）,asts用来生成AST词表
    nls = []
    asts = []
    ast_nums = []
    codes = []
    simple_name_list = []

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
        # 找出AST中的语义节点(变量名...)
        simple_names = traverse_simple_name(tree)[: max_simple_len]

        number = int(os.path.split(file)[1])
        #  number = int(file.split("\\\\")[-1])  # AST对应存储的文件名
        code_token = tokenize(code)[1:-1]  # 对源代码进行分词，本模型未使用，为了其他基线模型

        ast_num = str(number)

        ast_nums.append(ast_num)
        nls.append(seq)
        codes.append(code_token)
        asts.append(traverse_label(tree))
        simple_name_list.append(simple_names)

        """将AST树存储到tree目录下"""
        with open(output_data_dir + "/tree/" + set_name + "/" + ast_num, "wb", 1) as f:
            pickle.dump(tree, f)

    print("{} files skipped".format(skip))
    # 生成训练词表
    if set_name == 'train':
        # 1. 生成AST词表
        ast_counter = Counter([l for s in asts for l in s])
        ast_vocab = {i: w for i, w in enumerate(
            ["<PAD>", "<UNK>"] + sorted([x[0] for x in ast_counter.most_common(vocab_size) if x[1] > min_count]))}
        # 2. 生成注释词表
        nl_tokens = []
        for l in nls:
            for x in l:
                nl_tokens.append(x)

        nl_counter = Counter(nl_tokens)
        print(nl_counter['</s>'])
        nl_tokens = sorted(x[0] for x in nl_counter.most_common(vocab_size) if x[1] > min_count)

        nl_tokens += ['<SimpleName>_' + str(i) for i in range(max_simple_len)]
        nl_vocab = {i: w for i, w in enumerate(
            ["<PAD>", "<UNK>"] + nl_tokens)}
        # 3. 生成源代码词表（本模型未使用）
        code_counter = Counter([x for l in codes for x in l])
        code_vocab = {i: w for i, w in enumerate(
            ["<PAD>", "<UNK>"] + sorted([x[0] for x in code_counter.most_common(vocab_size) if x[1] > min_count]))}

        print('ast_vocab', list(ast_vocab.items()))
        print('nl_vocab', list(nl_vocab.items()))

        # 保存
        pickle.dump(ast_vocab, open(output_data_dir + "/ast_i2w.pkl", "wb"))
        pickle.dump(nl_vocab, open(output_data_dir + "/nl_i2w.pkl", "wb"))
        pickle.dump(code_vocab, open(output_data_dir + "/code_i2w.pkl", "wb"))
    else:
        nl_vocab = read_pickle(output_data_dir + "/nl_i2w.pkl")

    # 摘要中的词，如果不在词表中，也不在AST节点中，设置为UNK. 如果不在词表中，在simpleName中，替换为simpleName
    for i in range(len(ast_nums)):
        ast_num = ast_nums[i]
        nl = nls[i]
        code = codes[i]
        simple_names = simple_name_list[i]

        new_nl = []
        for nl_token in nl:
            if nl_token in nl_vocab.values():
                new_nl.append(nl_token)
            elif nl_token in simple_names:
                new_nl.append('<SimpleName>_' + str(simple_names.index(nl_token)))
            else:
                new_nl.append('<UNK>')

        outputs.append({'ast_num': ast_num,
                        'code': code,
                        'nl': new_nl})

    return outputs


def rebuild_tree(root):
    # 先反转根节点的子节点顺序，尽可能多的保留信息
    ls = list(reversed(root.children))
    root.children = ls

    for node in traverse(root):
        if "=" not in node.label and "(SimpleName)" in node.label:
            id_ = get_identifier(node.children[0].label)
            if id_ is None:
                raise Exception("ERROR!")

            node.is_simple_name = True
            # 将变量名保存到自身
            node.simple_name = id_.lower()

            # 将simpleName的子节点拆分为多个单词节点
            words = deal_with_simple_name(id_)
            node.children = [Node(label=w, parent=node, is_leaf_node=True, children=[]) for w in words]

        elif node.label[:6] == "value=":
            value_ = get_values(node.label)
            node.is_simple_name = True
            node.simple_name = value_.lower()
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
    s = s.strip()
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
    a = re.findall("[A-Z](?:[A-Z]+)", s)

    for i in a:
        s = s.replace(i[1:], i[1:].lower())
    pattern = "[A-Z._]"
    s = re.sub(pattern, lambda x: " " + x.group(0), s)
    words = []
    for x in s.split():
        word_token = tokenizer.tokenize(x)
        words.extend(word_token)
    return words


def tokenize(s):
    standard_tokens = ['<s>'] + tokenizer.tokenize(s) + ['</s>']
    return standard_tokens


def isnum(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


if __name__ == '__main__':
    process(data_dir='../../data', output_data_dir='../../data_set')
