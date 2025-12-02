#!/usr/bin/env python

"""
Advent of Code 2025, Day 2: Gift Shop



Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


def is_twice(i):
    s = str(i)
    if len(s) & 1: return False
    half = len(s)//2
    return s[:half] == s[half:]


def part1(ranges):
    sum_invalid = 0
    
    for begin, end in ranges:
        # print(f'{begin}-{end}')
        for i in range(begin, end+1):
            if is_twice(i):
                sum_invalid += i
                # print(f'  {i}')

    print(sum_invalid)


def part2(ranges):
    pass


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()
    def split_range(r):
        return tuple(int(x) for x in r.split('-'))
    
    ranges = [split_range(r) for r in input[0].split(',')]

    # read input as a Grid object, where each 
    # grid = grid_read(input_filename(), True)
  
    part1(ranges)
    part2(ranges)
