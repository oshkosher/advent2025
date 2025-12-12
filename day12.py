#!/usr/bin/env python3

"""
Advent of Code 2025, Day 12: Christmas Tree Farm

12x5: 1 0 1 0 2 2
width x length: quanitity of each shape
rotate and flip

all presents are 3x3 grids

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


class Region:
    def __init__(self, width, height, counts):
        self.width = width
        self.height = height
        self.counts = tuple(counts)

    def __repr__(self):
        return f'Region({self.width}x{self.height}, {" ".join([str(c) for c in self.counts])})'


class Present:
    def __init__(self, lines):
        """
        lines: 3 lines of 3 chars, '#' or '.'
        """
        self.original = lines[:]

        # all rotations and flips
        self.rots = []

        self.n_set = sum([row.count('#') for row in self.original])

        # fill self.rots with all rotations and flipped versions
        # of this pattern, with duplicates removed
        self.set_rots()

    def __len__(self):
        """
        Returns the number of set pixels
        """
        return self.n_set

    def set_rots(self):
        """
        Fill the self.rots array by finding all unique rotations and flips
        """
        rot_set = set()
        for flip_count in range(2):
            for rotate_count in range(4):
                rot_set.add(tuple(self.original))
                self.rotate()
            self.flip()

        self.rots = list(rot_set)

    def rotate(self):
        """
        Apply a 90 degree rotation clockwise
        """
        # my row 0 is pattern column 0, backwards
        height = len(self.original)
        width = len(self.original[0])
        
        rot = [[self.original[height-c-1][r] for c in range(height)]
               for r in range(width)]
        for r in range(width):
            self.original[r] = ''.join(rot[r])

    def flip(self):
        """
        Flip the pattern along the horizontal axis.
        """
        height = len(self.original)
        for r in range(height//2):
            tmp = self.original[r]
            self.original[r] = self.original[height-r-1]
            self.original[height-r-1] = tmp

    def print(self, pattern = None):
        if pattern == None:
            pattern = self.original
            
        for row in pattern:
            print(row)


def parse_input(input):
    line_no = 0

    presents = []
    
    while -1 == input[line_no].find('x'):
        line_no += 1
        start_line = line_no
        while len(input[line_no]) > 0:
            line_no += 1
        end_line = line_no

        present = Present(input[start_line:end_line])
        present.print()
        print(len(present))
        print()
        presents.append(present)

        line_no += 1

    regions = []
    
    while line_no < len(input):
        width, height, *counts = ints(input[line_no])
        regions.append(Region(width, height, counts))
        line_no += 1
        print(regions[-1])

    return present, regions


def part1(input):
    pass


def part2(input):
    pass


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()

    # read input as a Grid object, where each 
    # grid = grid_read(input_filename(), True)

    # p = Present(['.##', '###', '#.#'])
    # p.print()
    # print()
    
    # for variant in p.rots:
    #     p.print(variant)
    #     print()

    presents, regions = parse_input(input)

  
    # part1(input)
    # part2(input)
