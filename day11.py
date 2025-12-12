#!/usr/bin/env python3

"""
Advent of Code 2025, Day 11: Reactor

Counting paths through a graph

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


class Node:
    def __init__(self, name):
        self.name = name

        # outgoing nodes
        self.outs = []

        # incoming nodes
        self.ins = []

        # used in path counting, this is the number of paths leading
        # to this node from some starting point
        self.n_paths = 0

    @staticmethod
    def create_from_input_line(line):
        node = Node(line[:3])
        node.outs = line[5:].split()
        return node

    def __repr__(self):
        outs = ' '.join([o.name for o in self.outs])
        ins = ' '.join([o.name for o in self.ins])
        return f'Node({self.name} outs=({outs}) ins=({ins}))'

    # without these, it just use the pointer to the object, which should
    # be fine
    
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def has_all_n_paths(self):
        """
        Returns True if all the input nodes for this node have n_paths set.
        Without their n_paths set I can't compute my n_paths value.
        """
        for incoming in self.ins:
            if incoming.n_paths == None:
                return False
        return True

    def set_n_paths(self):
        """
        Sets my n_paths to be the sum of all my incoming nodes' n_paths.
        Return True if successful, False if any of them are net set.
        """
        if self.n_paths != None:
            return self.n_paths
        
        n_paths = 0
        for incoming in self.ins:
            if incoming.n_paths == None:
                return False
            n_paths += incoming.n_paths
        self.n_paths = n_paths
        
        return True


def parse_input(input):
    # name: Node
    node_map = {}

    # first just create an object for each name
    for line in input:
        name = line[:3]
        node = Node(name)
        node_map[name] = node
        
    # only listed as a destination, never a source
    node_map['out'] = Node('out')

    for line in input:
        src = node_map[line[:3]]

        # build src's 'outs' list
        src.outs = [node_map[d] for d in line[5:].split()]

        # add an entry to the destination node's 'ins' list
        for dest in src.outs:
            dest.ins.append(src)
            
    return node_map


def output_graphviz(node_map):
    print("""digraph day11 {
    svr [shape=rect];
    out [shape=doublecircle];
    node [style=filled];
    fft [shape=diamond fillcolor="blue"];
    dac [shape=diamond fillcolor="blue"];
    you [fillcolor="green"];
""")
    for node in node_map.values():
        for out in node.outs:
            print(f'{node.name} -> {out.name};')
    print('}')


def count_routes_dfs(node, dest, path_count = [0]):
    if node == dest:
        path_count[0] += 1
        return path_count[0]

    for child in node.outs:
        # print(f'{node.name} -> {child.name}')
        count_routes_dfs(child, dest, path_count)

    return path_count[0]


def part1(node_map):
    print(count_routes_dfs(node_map['you'], node_map['out']))


REPORT_FREQ = 10000000
next_report = REPORT_FREQ

l1_choke = set(['xed', 'kxy', 'nju'])
l2_choke = set(['yhv', 'oyz', 'qlj', 'uyb'])
l3_choke = set(['mfc', 'bjj', 'cin', 'xyw'])
l4_choke = set(['bid', 'jcy', 'cub', 'ooa'])
l5_choke = set(['you', 'tgk', 'cjp', 'ykg'])


def find_choke_points(src, dest, node_map):
    """
    Assuming there is a small set of nodes every path from src to dest
    pass through, 
    """
    pass


def part2_dft_too_slow(node, path_count, visited, depth=0):
    global next_report, REPORT_FREQ
    if node.name == 'out':
        # print('at out')
        if 'dac' in visited and 'fft' in visited:
            path_count[0] += 1
            if path_count[0] == next_report:
                print(f'{path_count[0]} paths found...')
                sys.stdout.flush()
                next_report += REPORT_FREQ
        return

    # prefix = ' ' * depth
    # print(f'{prefix}{node.name}')
    
    if node.name in visited:
        print(f'Cycle to {node}')
        return

    if node.name in l2_choke and 'fft' not in visited:
        return

    if node.name in l5_choke and 'dac' not in visited:
        return
        
    
    visited.add(node.name)

    for child in node.outs:
        # prefix = ' ' * depth
        # print(f'{prefix}{node.name} -> {child.name}')
        part2_dft_too_slow(child, path_count, visited, depth+1)

    visited.remove(node.name)


def reset_path_counts(node_map):
    for node in node_map.values():
        node.n_paths = 0


def find_descendent_subset(src):
    """
    Returns a set of all the nodes reachable when starting at src
    by doing a breadth-first traverse of 'outs' lists.
    """
    visited = set()

    visit_q = collections.deque()
    visit_q.append(src)
    while len(visit_q):
        node = visit_q.popleft()
        if node in visited:
            continue
        visited.add(node)

        for child in node.outs:
            visit_q.append(child)

    return visited


def find_ancestor_subset(dest):
    """
    Returns a set of all the nodes that can reach dest
    by doing a breadth-first traverse of 'ins' lists.    
    """
    visited = set()

    visit_q = collections.deque()
    visit_q.append(dest)
    while len(visit_q):
        node = visit_q.popleft()
        if node in visited:
            continue
        visited.add(node)

        for child in node.ins:
            visit_q.append(child)

    return visited


def create_subgraph(node_map, selected_set):
    """
    Returns a name->node map of just the nodes in the graph included
    in selected_set. This also remove entries from each node's 'ins' and
    'outs' lists that are not in selected_set.
    """
    sub_node_map = {
        n.name: Node(n.name) for n in node_map.values() if n in selected_set
    }

    # convert a node from node_map into a node in sub_node_map
    def to_sub(orig):
        return sub_node_map[orig.name]

    for node in sub_node_map.values():
        orig = node_map[node.name]
        node.outs = [to_sub(x) for x in orig.outs if x in selected_set]
        node.ins = [to_sub(x) for x in orig.ins if x in selected_set]

    return sub_node_map


def create_reachable_subgraph(src, dest):
    """
    returns a node_map consisting of just the nodes reachable from src
    new nodes are created, with 'ins' and 'outs' reflecting this subgraph.
    """

    print(f'creating reachable subgraph from {src.name}')
    
    # nodes visited
    visited = {}

    visit_q = collections.deque()
    visit_q.append(src)

    while len(visit_q):
        node = visit_q.popleft()
        if node.name not in visited:
            visited[node.name] = node
        for child in node.outs:
            visit_q.append(child)

    print(f'{len(visited)} nodes in subgraph')
    for node in visited.values():
        print(node)
            
    node_map = {}
    for name in visited:
        pass

    return visited
    


def count_paths_dp(src, dest):
    path_map = {}
    # path_map = {src: 1}
    
    q = collections.deque()
    q.append(src)

    def is_all_incoming_computed(node):
        for incoming in node.ins:
            if incoming not in path_map:
                return False
        return True

    while len(q):
        node = q.popleft()
        
        if node == src:
            path_count = 1
        else:
            path_count = sum([path_map[incoming] for incoming in node.ins])
        path_map[node] = path_count
        # print(f'{path_count} paths to {node.name}')

        for out in node.outs:
            if is_all_incoming_computed(out):
                q.append(out)
                    
    return path_map[dest]


def test_depths(node_map):
    """
    Check if every route from the root to a node is the same length.
    Answer: yes, many are reachable via paths of different lengths.
    """
    if len(node_map) < 100:
        root = node_map['you']
    else:
        root = node_map['svr']
        
    node_depths = {root: 0}
    visited = set()
    
    q = collections.deque()
    q.append(root)
    n_checked = 0

    while len(q):
        node = q.popleft()
        if node in visited:
            continue
        visited.add(node)
        n_checked += 1
        
        depth = node_depths[node]

        for child in node.outs:
            if child in node_depths:
                if node_depths[child] != depth + 1:
                    print(f'Inconsistency: {child} is at depths '
                          f'{node_depths[child]} and {depth+1}')
            else:
                node_depths[child] = depth + 1
            q.append(child)

    print(f'{n_checked} checked.')
    
    

def part2(input):
    """
    passes through dac and fft
    dac: sxv emj
    fft: yzt blx jpn xrt ysw
    paths from svr to out including dac and fft

    more than 5,560,000,000 paths
    """
    root = node_map['svr']
    fft = node_map['fft']
    dac = node_map['dac']
    out = node_map['out']
    # find_part2(node, path_count, visited)

    # routes_root_to_fft = count_paths_bfs(root, fft, node_map)
    # print(f'{routes_root_to_fft} routes from root to fft')

    # routes_dac_to_count = count_paths_bfs(dac, out, node_map)
    # print(f'{routes_dac_to_count} routes from dac to out')

    # n = count_paths_bfs(node_map['you'], node_map['out'], node_map)

    m = create_reachable_subgraph(node_map['you'])



if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()
    node_map = parse_input(input)

    # print(node_map['you'])

    # output_graphviz(node_map)

    # print(node_map['you'])
    # print(node_map['svr'])
    
    # part1(node_map)

    # test_depths(node_map)

    fft_in_set = find_ancestor_subset(node_map['fft'])
    fft_out_set = find_descendent_subset(node_map['fft'])
    dac_in_set = find_ancestor_subset(node_map['dac'])
    dac_out_set = find_descendent_subset(node_map['dac'])

    subgraph_fft_in = create_subgraph(node_map, fft_in_set)
    paths_to_fft = count_paths_dp(subgraph_fft_in['svr'],
                                  subgraph_fft_in['fft'])
    print(paths_to_fft)
    print()

    subgraph_fft_to_dac = create_subgraph(node_map, fft_out_set & dac_in_set)
    paths_fft_to_dac = count_paths_dp(subgraph_fft_to_dac['fft'],
                                      subgraph_fft_to_dac['dac'])
    print(paths_fft_to_dac)
    print()
    
    subgraph_dac_out = create_subgraph(node_map, dac_out_set)
    paths_dac_out = count_paths_dp(subgraph_dac_out['dac'],
                                   subgraph_dac_out['out'])
    print(paths_dac_out)
    print()
    print(paths_to_fft * paths_fft_to_dac * paths_dac_out)
