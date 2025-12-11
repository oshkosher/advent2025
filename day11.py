#!/usr/bin/env python3

"""
Advent of Code 2025, Day 11: Reactor


Start at "you", end at "out"
Is a DAG (no cycles)


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
    def __init__(self, line):
        self.name = line[:3]

        # outgoing nodes
        # initially store the names of each outgoing node, then when
        # all nodes have been read, replaced those with node objects.
        self.outs = line[5:].split()

        # incoming nodes
        # initially a set of names, then changed to a list of objects
        self.ins = set()

    def __str__(self):
        outs = ' '.join([o.name for o in self.outs])
        ins = ' '.join([o.name for o in self.ins])
        return f'Node({self.name} outs=({outs}) ins=({ins}))'


def parse_input(input):
    node_list = [Node(line) for line in input]
    node_map = {n.name: n for n in node_list}

    # only listed as a destination, never a source
    node_map['out'] = Node('out: ')
    
    for src in node_list:
        src.outs = [node_map[dest_name] for dest_name in src.outs]

        for dest in src.outs:
            dest.ins.add(src.name)

    for node in node_list:
        node.ins = [node_map[name] for name in node.ins]
            
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
          


def count_routes(node, dest, path_count = [0]):
    if node == dest:
        path_count[0] += 1
        return path_count[0]

    for child in node.outs:
        # print(f'{node.name} -> {child.name}')
        count_routes(child, dest, path_count)

    return path_count[0]


def part1(node_map):
    print(count_routes(node_map['you'], node_map['out']))


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


def find_part2(node, path_count, visited, depth=0):
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
        find_part2(child, path_count, visited, depth+1)

    visited.remove(node.name)
    
    

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

    routes_to_fft = count_routes(root, fft)
    print(f'{routes_to_fft} routes from root to fft')

    routes_from_dac = count_routes(dac, out)
    print(f'{routes_from_dac} routes from dac to out')



if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()
    node_map = parse_input(input)

    # print(node_map['you'])

    # output_graphviz(node_map)

    # print(node_map['you'])
    # print(node_map['svr'])
    
    part1(node_map)
    
    # part2(node_map)
