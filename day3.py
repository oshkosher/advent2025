#!/usr/bin/env python

"""
Advent of Code 2025, Day 3: Lobby

Subset of digits maximizing value

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


def max_joltage(digits, n_digits):
    """
    A greedy algorithm is sufficient for this.
    At each step, choose the highest digit in the available range.
    
    The end of the available range leaves at least one digits for each
    remaining digit needed. For example, when choosing the first of 12
    digits, exclude the last 11 digits of the input list.

    The beginning of the available range starts at the beginning of the
    input list, then after a digit is chosen, the next digit is the
    beginning of the available range.
    """
    
    joltage = 0
    start_pos = 0
    for di in range(1, n_digits+1):
        # if we still need to pick k digits after this one, skip the
        # last k elements of digits[]
        search_end = len(digits) - n_digits + di

        # find the best digits in our search range
        best_d = max(digits[start_pos:search_end])
        # print(f'  best in digits[{start_pos}:{search_end}]: {best_d}')

        # find the first place this digit appears in our given search range
        start_pos = digits.index(best_d, start_pos) + 1

        # build the joltage value, adding a new least-significant digit
        joltage = 10*joltage + best_d

    return joltage


def sum_max_joltages(input, digit_count):
    joltage_sum = 0
    
    for line in input:
        digits = [int(c) for c in line]
        max_j = max_joltage(digits, digit_count)
        joltage_sum += max_j

    print(joltage_sum)


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()

    sum_max_joltages(input, 2)
    sum_max_joltages(input, 12)
