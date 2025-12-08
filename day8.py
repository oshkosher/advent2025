#!/usr/bin/env python3

"""
Advent of Code 2025, Day 8: Playground

Merging connected subgraphs

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools
from typing import Optional

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


class Box:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.circuit: Optional[Circuit] = None

    def __repr__(self):
        c = self.circuit.idx if self.circuit else 'None'
        return f'Box({self.x}, {self.y}, {self.z}, circuit={c})'

    def key(self):
        return (self.x, self.y, self.z)

    # define __hash__ and __eq__ so boxes can be put into a set()
    
    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        return self.key() == other.key()


class Circuit:
    # number of Circuit objects created
    count = 0

    # list of all Circuit objects
    circuit_list: list[Circuit] = []
    
    def __init__(self, idx):
        # assign indices to Circuit objects so when two Circuits are merged
        # there's an easy way to choose which Circuit will contain all
        # the boxes
        self.idx = idx
        self.boxes = set()

    def __repr__(self):
        return f'Circuit({self.idx}, {len(self.boxes)} boxes)'

    def __len__(self):
        return len(self.boxes)

    # order by size
    def __lt__(self, other):
        return len(self) > len(other)

    @staticmethod
    def create():
        c = Circuit(Circuit.count)
        Circuit.count += 1
        Circuit.circuit_list.append(c)
        return c

    def add(self, box):
        box.circuit = self
        self.boxes.add(box)

    def remove(self, box):
        box.circuit = None
        self.boxes.remove(box)


def distance_sq(a : Box, b : Box):
    """
    Euclidean distance between two boxes, squared.
    This problem only compares distances, so there's no need to introduce
    errors doing a square root calculation.
    """
    return abs(a.x - b.x)**2 + abs(a.y - b.y)**2 + abs(a.z - b.z)**2


def connect(box1 : Box, box2 : Box):
    if box1.circuit == None:
        if box2.circuit == None:
            # neither box is in a circuit; create a new circuit
            new_circuit = Circuit.create()
            new_circuit.add(box1)
            new_circuit.add(box2)
            #print(f'join {box1} and {box2}')
        else:
            # box2 is in a circuit, box1 is not. Add box1 to box2's circuit
            box2.circuit.add(box1)
            #print(f'add {box1} to {box2}')
    else:
        if box2.circuit == None:
            # box1 is in a circuit, box2 is not. Add box2 to box1's circuit
            box1.circuit.add(box2)
            #print(f'add {box2} to {box1}')
        else:

            # both boxes are already in circuits
            # figure out which circuit has the lower index, and move all the
            # boxes from the other circuit into it
            
            if box1.circuit == box2.circuit:
                # print(f'already in the same circuit {box1} {box2}')
                return
            
            # dest is the circuit with the lower index
            if box1.circuit.idx < box2.circuit.idx:
                src, dest = box2.circuit, box1.circuit
            else:
                src, dest = box1.circuit, box2.circuit
            # print(f'move boxes from {src} to {dest}')

            # need to make a copy of src.boxes because we'll be
            # modifying src.boxes
            to_move = list(src.boxes)
            for box in to_move:
                src.remove(box)
                dest.add(box)

            # print(f'  {src}')
            # print(f'  {dest}')


def parse_input(lines) -> list[Box]:
    def to_box(line):
        return Box(*[int(x) for x in line.split(',')])

    return [to_box(line) for line in lines]


def pairs_by_distance(boxes):
    # [(distance_squared, box1, box2), ...]
    dist_list = []
    for i, box1 in enumerate(boxes):
        for j in range(i+1, len(boxes)):
            box2 = boxes[j]
            dist_list.append((distance_sq(box1, box2), box1, box2))
    dist_list.sort()
    return dist_list


def mult(lst):
    p = 1
    for x in lst:
        p *= x
    return p


def part1(boxes):
    # [(distance_squared, box1, box2), ...]
    dist_list = pairs_by_distance(boxes)

    # automatically distinguish between sample and real input
    connect_count = 10 if len(boxes) < 100 else 1000
    
    for _, box1, box2 in dist_list[:connect_count]:
        # print(f'join {box1} and {box2}')
        connect(box1, box2)

    circuits_by_size = Circuit.circuit_list[:]
    circuits_by_size.sort()
    print(mult([len(c) for c in circuits_by_size[:3]]))

    # pass the rest of the box pairs to part2
    return dist_list, connect_count


def part2(boxes, dist_list, connect_count):
    ci = connect_count  # connection index; next connetion to make
    largest_circuit = 0

    # keep making connections until all boxes are in one circuit
    while True:
        _, box1, box2 = dist_list[ci]
        ci += 1

        connect(box1, box2)

        # if len(box1.circuit) > largest_circuit:
        #     print(f'[{ci}] have circuit of size {len(box1.circuit)}')
        #     largest_circuit = len(box1.circuit)
        
        if len(box1.circuit) == len(boxes):
            # print(f'{box1} and {box2}, {box1.circuit}')
            print(box1.x * box2.x)
            break


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()

    boxes = parse_input(input)
    print(boxes[0])
  
    dist_list, connect_count = part1(boxes)
    part2(boxes, dist_list, connect_count)
