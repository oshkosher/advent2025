#!/usr/bin/env python3

"""
Advent of Code 2025, Day 8: Playground

Merging connected subgraphs

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools, heapq
from typing import Optional

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *

"""
I used the Disjoint Set Union algorithm from here:
  https://cp-algorithms.com/data_structures/disjoint_set_union.html

Each set is represented as a tree, where each node maintains a pointer
to its parent, and the root points to itself. To check if two nodes are
in the same set, chase the parent pointers from each and see if they end
up at the same root. As an optimization, after chasing parent pointers
in this operation, modify every visited node's parent pointer to point
directly to the root, so the next search will be quicker.

To merge two sets, just graft one tree onto another by modifying the
root's parent pointer.

The set operations end up being very quick in this problem. Most of the
time is spent building the heap of all possible pairs.
  all-pairs heap: 443 ms
  part1: 5 ms
  part2: 30 ms
"""

class Box:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent: Box = self

        # circuit size
        self.csize: int = 1

    def __repr__(self):
        return f'Box({self.x}, {self.y}, {self.z}, circuit={self.find_root().key()}, csize={self.csize})'

    def find_root(self):
        """
        todo: after traversing, attach everyone directly to the root
        """
        if self.parent == self: return self
        
        traversed = [self]
        p = self.parent
        while p != p.parent:
            traversed.append(p)
            p = p.parent
        for box in traversed:
            box.parent = p
        return p

    def key(self):
        return (self.x, self.y, self.z)

    def __lt__(self, other):
        return self.csize < other.csize


PairDistance = collections.namedtuple('PairDistance', ['dist', 'box1', 'box2'])


def distance_sq(a : Box, b : Box):
    """
    Euclidean distance between two boxes, squared.
    This problem only compares distances, so there's no need to introduce
    errors doing a square root calculation.
    """
    x = a.x - b.x
    y = a.y - b.y
    z = a.z - b.z
    return x*x + y*y + z*z
    # return abs(a.x - b.x)**2 + abs(a.y - b.y)**2 + abs(a.z - b.z)**2


def connect(box1 : Box, box2 : Box):
    # print(f'connect {box1} and {box2}')
    set1 = box1.find_root()
    set2 = box2.find_root()
    # print(f'  circuits {box1.key()}, {box2.key()}')
    if set1 != set2:
        # attach the smaller set to the larger set
        if set1.csize < set2.csize:
            small, large = set1, set2
        else:
            large, small = set1, set2
        small.parent = large
        large.csize += small.csize
        # print(f'  large={large.key()},size={large.csize}')
            

def parse_input(lines: list[str]) -> list[Box]:
    def to_box(line):
        return Box(*[int(x) for x in line.split(',')])

    return [to_box(line) for line in lines]


def create_pair_distance_queue(boxes: list[Box]) -> list[PairDistance]:
    """
    Create a heap containing a PairDistance object for every pair of boxes.
    This holds pointers to both boxes and the distance between them.
    The heap is a min-heap, so pair are pulled shortest-first.
    """
    dist_q: list[PairDistance] = []

    for i in range(len(boxes)-1):
        box1 = boxes[i]
        for j in range(i+1, len(boxes)):
            box2 = boxes[j]
            pd = PairDistance(distance_sq(box1, box2), box1, box2)
            dist_q.append(pd)

    heapq.heapify(dist_q)
    return dist_q


def part1(boxes: list[Box], dist_q: list[PairDistance]):
    # automatically distinguish between sample and real input
    count = 10 if len(boxes) < 100 else 1000

    for _ in range(count):
        pair = heapq.heappop(dist_q)
        connect(pair.box1, pair.box2)
    
    # order the root nodes by decreasing set size
    circuits_by_size = [box for box in boxes if box.parent == box]
    circuits_by_size.sort(reverse=True)
    
    print(mult([c.csize for c in circuits_by_size[:3]]))


def part2(boxes: list[Box], dist_q: list[PairDistance]):
    # keep making connections until all boxes are in one circuit
    while True:
        pair = heapq.heappop(dist_q)
        box1 = pair.box1
        box2 = pair.box2
        connect(box1, box2)
        
        circuit = box1.find_root()
        
        if circuit.csize == len(boxes):
            # print(f'{box1} and {box2}, {box1.circuit}')
            print(box1.x * box2.x)
            break


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()

    boxes = parse_input(input)

    timer = time.perf_counter_ns()
    dist_q = create_pair_distance_queue(boxes)
    timer = time.perf_counter_ns() - timer
    # print(f'dist_q {timer/1e6:.0f} ms')
    
    t0 = time.perf_counter_ns()
    part1(boxes, dist_q)
    t1 = time.perf_counter_ns()
    part2(boxes, dist_q)
    t2 = time.perf_counter_ns()
    # print(f'part1 {(t1-t0)/1_000_000:.0f} millis')
    # print(f'part2 {(t2-t1)/1_000_000:.0f} millis')
