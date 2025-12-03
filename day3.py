#!/usr/bin/env python

"""
Advent of Code 2025, Day 3: Lobby



Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


def max_joltage_2(digits):
    d1 = max(digits[:-1])
    d1_idx = digits.index(d1)
    d2 = max(digits[d1_idx+1:])
    return d1 * 10 + d2


def max_joltage(digits, n_digits):
    joltage = 0
    start_pos = 0
    for di in range(1, n_digits+1):
        # if we still need to pick k digits after this one, skip the
        # last k elements of digits[]
        search_end = len(digits) - n_digits + di
        best_d = max(digits[start_pos:search_end])
        # print(f'  best in digits[{start_pos}:{search_end}]: {best_d}')
        start_pos = digits.index(best_d, start_pos) + 1
        joltage = 10*joltage + best_d

    return joltage


def part1(input):
    joltage_sum = 0
    
    for line in input:
        digits = [int(c) for c in line]
        max_j = max_joltage_2(digits)
        # print(f'{line}: {max_j}')
        joltage_sum += max_j

    print(joltage_sum)


def part2(input):
    joltage_sum = 0
    
    for line in input:
        # print(line)
        digits = [int(c) for c in line]
        max_j = max_joltage(digits, 12)
        # print(f'{line}: {max_j}')
        joltage_sum += max_j

    print(joltage_sum)


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()

    # read input as a Grid object, where each 
    # grid = grid_read(input_filename(), True)
  
    part1(input)
    part2(input)
