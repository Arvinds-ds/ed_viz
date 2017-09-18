import re
import uuid
import graphviz as gv
from graphviz import Digraph
import tensorflow as tf
import edward as ed


graph_pref = {
    'fontcolor': '#414141',
    'style': 'rounded',
    'pack':'True'
}

name_scope_graph_pref = {
    'bgcolor': '#eeeeee',
    'color': '#aaaaaa',
    'penwidth': '2',
}

non_name_scope_graph_pref = {
    'fillcolor':  'white',
    'color': 'white',
}

node_pref = {
    'style': 'filled',
    'fillcolor': '#eeeeee',
    'shape': 'ellipse',
    'color': '#aaaaaa',
    'penwidth': '2',
    'fontcolor': '#414141',
    'fontsize':'8'
}

edge_pref = {
    'color': '#aaaaaa',
    'arrowsize': '1.2',
    'penwidth': '1',
    'fontcolor': '#414141',
}

rv_sample_style = {
        'fontname': 'Helvetica',
        'shape': 'square',
        'fontcolor': 'white',
        'color': 'white',
        'style': 'filled',
        'fillcolor': '#006699',
        'fontsize':'12'
}

rv_style = {
        'fontname': 'Helvetica',
        'shape': 'circle',
        'fontcolor': 'white',
        'color': 'blueviolet',
        'style': 'filled',
        'fillcolor': 'blueviolet',
        'fontsize':'12'
}

# index of subgraph
CLUSTER_INDEX = 0


def tf_digraph(name=None, name_scope=None, style=True):
    """
    Return graphviz.dot.Digraph with TensorBoard-like style.
    @param  name
    @param  name_scope
    @param  style
    @return graphviz.dot.Digraph object
    """
    digraph = gv.Digraph(name=name)
    if name_scope:
        digraph.graph_attr['label'] = name_scope
    if style is False: return digraph

    if name_scope:
        digraph.graph_attr.update(name_scope_graph_pref)
    else:
        digraph.graph_attr.update(non_name_scope_graph_pref)
    digraph.graph_attr.update(graph_pref)
    digraph.node_attr.update(node_pref)
    digraph.edge_attr.update(edge_pref)
    return digraph


def nested_dict(dict_, keys, val):
    """
    Assign value to dictionary.
    @param  dict_
    @param  keys
    @param  val
    @return dictionary
    """
    cloned = dict_.copy()
    if len(keys) == 1:
        cloned[keys[0]] = val
        return cloned
    dd = cloned[keys[0]]
    for k in keys[1:len(keys)-1]:
        dd = dd[k]
    last_key = keys[len(keys)-1]
    dd[last_key] = val
    return cloned


def node_abs_paths(node):
    """
    Return absolute node path name.
    @param  node
    @return string
    """
    node_names = node.name.split('/')
    return ['/'.join(node_names[0:i+1]) for i in range(len(node_names))]

def get_rv_nodes(tfgraph, depth=2):
    """
    Return dictionary of node.
    @param  tfgraph
    @param  depth
    @return dictionary
    """
    table = {}
    max_depth = depth
    ops = [n for n in tfgraph.get_operations() if 'sample' in n.name]
    for depth_i in range(max_depth):
        for op in ops:
            abs_paths = node_abs_paths(op)
            if depth_i >= len(abs_paths): continue
            ps = abs_paths[:depth_i+1]
            if len(ps) == 1:
                key = '/'.join(abs_paths[0:depth_i+1])
                if not key in table: table[key] = {}
            else:
                table = nested_dict(table, ps, {})
    rvs = []
    for scope in table.keys():
        rvs.extend(list(table[scope].keys()))
    return rvs

def node_table(tfgraph, depth=1):
    """
    Return dictionary of node.
    @param  tfgraph
    @param  depth
    @return dictionary
    """
    table = {}
    max_depth = depth
    ops = tfgraph.get_operations()
    for depth_i in range(max_depth):
        for op in ops:
            abs_paths = node_abs_paths(op)
            if depth_i >= len(abs_paths): continue
            ps = abs_paths[:depth_i+1]
            if len(ps) == 1:
                key = '/'.join(abs_paths[0:depth_i+1])
                if not key in table: table[key] = {}
            else:
                table = nested_dict(table, ps, {})
    return table


def node_shape(tfnode, depth=1):
    """
    Return node and the children.
    @param  tfnode
    @param  depth
    @return string, list
    """
    outpt_name = tfnode.name
    if len(outpt_name.split('/')) < depth: return None
    on = '/'.join(outpt_name.split('/')[:depth]) # output node
    result = re.match(r"(.*):\d*$", on)
    if not result: return None
    on = result.groups()[0]
    if tfnode.shape.ndims is None:
        return on, []
    else:
        return on, tfnode.shape.as_list()


def node_input_table(tfgraph, depth=1):
    """
    Return table of operations
    @param  tfgraph
    @param  depth
    @return dictionary, table of operations
    """
    table = {}
    inpt_op_table = {}
    inpt_op_shape_table = {}
    ops = [n for n in tfgraph.get_operations() if 'sample' in n.name]
    for op in tfgraph.get_operations():
        op_name = op.name.split('/')[0:depth]
        opn = '/'.join(op_name)
        if not opn in inpt_op_table:
            inpt_op_table[opn] = []
        inpt_op_list = ['/'.join(inpt_op.split('/')[0:depth]) for inpt_op in op.node_def.input]
        inpt_op_table[opn].append(inpt_op_list)
        for output in op.outputs:
            for i in range(depth):
                shape = node_shape(output, depth=i+1)
                if shape: inpt_op_shape_table[shape[0]] = shape[1]
    for opn in inpt_op_table.keys():
        t_l = []
        for ll in inpt_op_table[opn]:
            list.extend(t_l, ll)
        table[opn] = list(set(t_l))
    return table, inpt_op_shape_table


def add_nodes(node_table, name=None, name_scope=None, style=True):
    """
    Add TensorFlow graph's nodes to graphviz.dot.Digraph.
    @param  node_table
    @param  name
    @param  name_scope
    @param  style
    @return graphviz.dot.Digraph object
    """
    global CLUSTER_INDEX
    if name:
        digraph = tf_digraph(name=name, name_scope=name_scope, style=style)
    else:
        digraph = tf_digraph(name=str(uuid.uuid4().get_hex().upper()[0:6]), name_scope=name_scope, style=style)
    graphs = []
    for key, value in node_table.items():
        if len(value) > 0:
            sg = add_nodes(value, name='cluster_%i' % CLUSTER_INDEX, name_scope=key.split('/')[-1], style=style)
            sg.node(key, key.split('/')[-1])
            CLUSTER_INDEX += 1
            graphs.append(sg)
        else:
            if is_rv(key):
                digraph.node(key, key.split('/')[-1],_attributes=rv_style)
            else:
                digraph.node(key, key.split('/')[-1])
    for tg in graphs:
        digraph.subgraph(tg)
    return digraph


def edge_label(shape):
    """
    Returen texts of graph's edges.
    @param  shape
    @return
    """
    if len(shape) == 0: return ''
    if shape[0] is None: label = "?"
    else: label = "%i" % shape[0]
    for s in shape[1:]:
        if s is None: label += "×?"
        else: label += u"×%i" % s
    return label

def get_distribution(name):
    for node in ed.random_variables():
        shortened_name = ''.join(node.name.split('_')[:-1])
        if shortened_name in name:
            return str(type(node).__name__)

def is_rv(name):
    for node in ed.random_variables():
        if node.name == name:
            return True
    return False
    
def is_rv_sample(name):
    for node in ed.random_variables():
        name_list = node.name.split('_')
        prefix_name = ''.join(name_list[:-1])
        suffix = name_list[-1]
        try:
            suffix_int = int(suffix.strip().strip('/'))
            if suffix_int > 1:
                check_name = prefix_name + '_' + str(suffix_int-1)
            else:
                check_name = prefix_name
            if name == check_name or name == check_name + '/':
                return True
        except:
            print("Failed in getting details for {}".format(node))
            
    return False

def is_rv1(name):
    for node in ed.random_variables():
        name_list = node.name.split('_')
        prefix_name = ''.join(name_list[:-1])
        suffix = name_list[-1]
        try:
            suffix_int = int(suffix.strip().strip('/'))
            if suffix_int > 2:
                check_name = prefix_name + '_' + str(suffix_int-2)
            else:
                check_name = prefix_name
            if name == check_name or name == check_name + '/':
                return True
        except:
            print("Failed in getting details for {}".format(node))
            
    return False
    

def rvs():
    rv_list, rv_sample_list = [],[]
    for node in ed.random_variables():
        shortened_name = ''.join(node.name.split('_')[:-1])
        rv_list.append(shortened_name)
        rv_sample_list.append(shortened_name + '_1')
    return rv_list, rv_sample_list

def add_edges(digraph, node_inpt_table, node_inpt_shape_table):
    """
    Add TensorFlow graph's edges to graphviz.dot.Digraph.
    @param  dirgraph
    @param  node_inpt_table
    @param  node_inpt_shape_table
    @return  graphviz.dot.Digraph
    """
    for node, node_inputs in node_inpt_table.items():
        if re.match(r"\^", node): continue
        for ni in node_inputs:
            if ni == node: continue
            if re.match(r"\^", ni): continue
            if not ni in node_inpt_shape_table:
                if is_rv(node):
                    digraph.edge(ni, node, label=' ~ ' + get_distribution(node))
                else:
                    digraph.edge(ni, node)
            else:
                shape = node_inpt_shape_table[ni]
                digraph.edge(ni, node, label=edge_label(shape))
    return digraph


def visualize_full(tfgraph=None, depth=2, name='G', style=True):
    """
    Return graphviz.dot.Digraph object with TensorFlow's Graphs.
    @param  depth
    @param  name
    @param  style
    @return  graphviz.dot.Digraph
    """
    global CLUSTER_INDEX
    CLUSTER_INDEX = 0
    if tfgraph is None:
        tfgraph = tf.get_default_graph()
    _node_table = node_table(tfgraph, depth=depth)
    _node_inpt_table, _node_inpt_shape_table = node_input_table(tfgraph, depth=depth)
    digraph = add_nodes(_node_table, name=name, style=style)
    digraph = add_edges(digraph, _node_inpt_table, _node_inpt_shape_table)
    return digraph



def visualize_simple():
    digraph = gv.Digraph(name='simple')
    digraph.graph_attr.update(graph_pref)
    digraph.node_attr.update(node_pref)
    digraph.edge_attr.update(edge_pref)    
    for node in ed.random_variables():
        digraph.node(node.unique_name, node.name + ' ~ ' + type(node).__name__ + '\n' + str(node.shape))
    for node in ed.random_variables():
        if node.get_parents() != []:
            for parent in node.get_parents():
                digraph.edge(parent.unique_name, node.unique_name)
    return digraph