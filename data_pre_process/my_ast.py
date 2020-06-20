import json
import re
import pickle
import torch


class Node:
    """树节点抽象"""
    def __init__(self, label="", parent=None, is_simple_name=False, simple_name=None, is_leaf_node=False, children=[]):
        self.label = label
        self.parent = parent
        self.children = children
        self.is_simple_name = is_simple_name
        self.simple_name = simple_name
        self.is_leaf_node = is_leaf_node


def parse(path):

    with open(path, "r") as f:
        try:
            num_objects = f.readline()
            nodes = [Node(children=[]) for i in range(int(num_objects))]
            for i in range(int(num_objects)):
                label = " ".join(f.readline().split(" ")[1:])[:-1]
                nodes[i].label = label
            while 1:
                line = f.readline()
                if line == "\n":
                    break
                p, c = map(int, line.split(" "))
                nodes[p].children.append(nodes[c])
                nodes[c].parent = nodes[p]

            data = json.loads(f.readline()[:-1])
            code = data['code']
            nl = data['comment']
        except Exception as e:
            print(e)
            return None, None, None
    return nodes[0], code, nl


def traverse(root):
    """traverse all nodes"""
    res = [root]
    for child in root.children:
        res = res + traverse(child)
    return res


def traverse_label(root):
    """return list of tokens"""
    li = [root.label]
    for child in root.children:
        li += traverse_label(child)
    return li


def get_method_name(root):
    for c in root.children:
        if c.label == "name (SimpleName)":
            return c.children[0].label[12:-1]


def traverse_simple_name(root):
    if root.is_simple_name:
        li = [root.simple_name]
    else:
        li = []
    for child in root.children:
        li += traverse_simple_name(child)
    return li


def get_bracket(s):
    if "value=" == s[:6] or "identifier=" in s[:11]:
        return None
    p = "\(.+?\)"
    res = re.findall(p, s)
    if len(res) == 1:
        return res[0]
    return s


def get_identifier(s):
    if "identifier=" == s[:11]:
        # 去除上引号和下引号
        return s[12:-1]
    else:
        return None


def get_values(s):
    if "value=" == s[:6]:
        # 去除上引号和下引号
        return s[7:-1]
    else:
        return None


def read_pickle(path):
    return pickle.load(open(path, "rb"))


def traverse_tree_to_generate_matrix(root_node, max_size, k, max_simple_len):
    # 先反转根节点的子节点顺序，尽可能多的保留信息，比如方法名，方法参数...
    ls = list(reversed(root_node.children))
    root_node.children = ls

    # 按照深度优先遍历取出前max_size个节点
    sub_tree(root_node, max_size=max_size)

    # 生成父子关系和兄弟关系矩阵
    root_id = root_node.num

    seq = [''] * max_size
    relative_parent_ids = torch.zeros((max_size, max_size))
    relative_brother_ids = torch.zeros((max_size, max_size))
    relative_semantic_ids = torch.zeros((max_size, max_size))
    semantic_convert_matrix = torch.zeros(max_size, max_simple_len)
    semantic_mask = torch.zeros(max_size)

    parent_map = {}
    brother_map = {}
    semantic_ids = []

    queue = [root_node]

    while queue:
        current_node = queue.pop()
        node_id = current_node.num
        seq[node_id] = current_node.label
        if current_node.is_simple_name:
            if len(semantic_ids) >= max_simple_len:
                continue
            semantic_ids.append(node_id)

        if node_id == root_id:
            parent_map[node_id] = [node_id]
            brother_map[node_id] = [node_id]

        if len(current_node.children) > 0:
            brother_node_ids = [x.num for x in current_node.children if x.num < max_size]
            for child in reversed(current_node.children):
                if child.num >= max_size:
                    continue
                child_id = child.num
                queue.append(child)

                parent_map[child_id] = parent_map[node_id] + [child_id]
                brother_map[child_id] = brother_node_ids

    for i in range(max_size):
        for j in range(max_size):
            if i not in parent_map or j not in parent_map:
                """忽略掉填充节点"""
                continue
            if i in parent_map[j] or j in parent_map[i]:
                """存在父子关系"""
                key = i if j in parent_map[i] else j
                rp = parent_map[key].index(j) - parent_map[key].index(i)
                rp = relative_range_map(rp, k)
                relative_parent_ids[i][j] = rp
            if i in brother_map[j]:
                rp = brother_map[j].index(j) - brother_map[j].index(i)
                rp = relative_range_map(rp, k)
                relative_brother_ids[i][j] = rp
            if i in semantic_ids and j in semantic_ids:
                rs = semantic_ids.index(j) - semantic_ids.index(i)
                rs = relative_range_map(rs, k)
                relative_semantic_ids[i][j] = rs

    for i, semantic_id in enumerate(semantic_ids):
        semantic_convert_matrix[semantic_id][i] = 1
        semantic_mask[semantic_id] = 1

    return seq, relative_parent_ids.long(), relative_brother_ids.long(), relative_semantic_ids.long(), semantic_convert_matrix, semantic_mask


def relative_range_map(value, k):
    """
    map value from [-k, k] to [1, 2k+1]
    """
    return max(-k, min(k, value)) + k + 1


def sub_tree(root_node, i=0, max_size=200):
    """
        树的最大节点个数不超过200
        """
    root_node.num = i
    i = i + 1
    if i > max_size:
        return -1
    else:
        for j, child in enumerate(root_node.children):
            i = sub_tree(child, i, max_size)
            if i == -1:
                root_node.children = root_node.children[:j]
                return -2
            if i == -2:
                root_node.children = root_node.children[:j + 1]
                return i
        return i
