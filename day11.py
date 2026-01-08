#!/usr/bin/env python3

"""
Advent of Code 2025, Day 11: Reactor

Counting paths through a graph

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools
from collections import deque

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

        # my index in the topological order
        self.topo_order = -1

    def __repr__(self):
        outs = ' '.join([o.name for o in self.outs])
        ins = ' '.join([o.name for o in self.ins])
        return f'Node({self.name} outs=({outs}) ins=({ins}))'

    def has_missing_inputs(self):
        """
        Returns True iff any of my input nodes do not have .topo_order set.
        """
        for parent in self.ins:
            if parent.topo_order == -1:
                return True
        return False
    

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


def topo_sort(node_map):
    """
    Returns a list of the nodes in topological order such that all a node's
    incoming nodes are earlier in the list.
    Also sets 'topo_order' field on each node object such that
      topo_order[i].topo_order == i
    """
    q = deque()
    topo_order = []

    # first enqueue every node with no inputs (should be just 'svr')
    for node in node_map.values():
        if len(node.ins) == 0:
            q.append(node)

    while q:
        node = q.popleft()
        node.topo_order = len(topo_order)
        topo_order.append(node)
        for child in node.outs:
            if not child.has_missing_inputs():
                q.append(child)
    """
    def name_and_order(node):
        return f'{node.name}({node.topo_order})'
                
    for i, node in enumerate(topo_order):
        parent_list = ' '.join([name_and_order(p) for p in node.ins])
        for parent in node.ins:
            assert (parent.topo_order != -1
                    and parent.topo_order < node.topo_order)
        print(f'{i}. {name_and_order(node)}: {parent_list}')
    """

    return topo_order


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


def count_paths_dp(topo_order, src, dest):
    # reset path counters
    for node in topo_order:
        node.n_paths = 0

    src.n_paths = 1

    for node in topo_order[src.topo_order+1 : dest.topo_order+1]:
        for parent in node.ins:
            node.n_paths += parent.n_paths

    return dest.n_paths


def part2(node_map, topo_order):
    # count all the paths from svr to fft
    paths_to_fft = count_paths_dp(
        topo_order, node_map['svr'], node_map['fft'])

    # count all the paths from fft to dac
    paths_fft_to_dac = count_paths_dp(
        topo_order, node_map['fft'], node_map['dac'])
        
    # count all the paths from dac to out
    paths_dac_out = count_paths_dp(
        topo_order, node_map['dac'], node_map['out'])

    # print(f'{paths_to_fft} x {paths_fft_to_dac} x {paths_dac_out}')
    
    # total possible = product of all three
    print(paths_to_fft * paths_fft_to_dac * paths_dac_out)


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()
    node_map = parse_input(input)
    topo_order = topo_sort(node_map)
    
    # output_graphviz(node_map)
    
    part1(node_map)
    part2(node_map, topo_order)
