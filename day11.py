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

    def __repr__(self):
        outs = ' '.join([o.name for o in self.outs])
        ins = ' '.join([o.name for o in self.ins])
        return f'Node({self.name} outs=({outs}) ins=({ins}))'


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
    

def part2(node_map):
    fft_in_set = find_ancestor_subset(node_map['fft'])
    fft_out_set = find_descendent_subset(node_map['fft'])
    dac_in_set = find_ancestor_subset(node_map['dac'])
    dac_out_set = find_descendent_subset(node_map['dac'])

    # count all the paths from svr to fft
    subgraph_fft_in = create_subgraph(node_map, fft_in_set)
    paths_to_fft = count_paths_dp(subgraph_fft_in['svr'],
                                  subgraph_fft_in['fft'])

    # count all the paths from fft to dac
    subgraph_fft_to_dac = create_subgraph(node_map, fft_out_set & dac_in_set)
    paths_fft_to_dac = count_paths_dp(subgraph_fft_to_dac['fft'],
                                      subgraph_fft_to_dac['dac'])

    # count all the paths from dac to out
    subgraph_dac_out = create_subgraph(node_map, dac_out_set)
    paths_dac_out = count_paths_dp(subgraph_dac_out['dac'],
                                   subgraph_dac_out['out'])

    # total possible = product of all three
    print(paths_to_fft * paths_fft_to_dac * paths_dac_out)


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()
    node_map = parse_input(input)

    # output_graphviz(node_map)
    
    part1(node_map)
    part2(node_map)
