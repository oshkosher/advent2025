#!/usr/bin/env python3

"""
Advent of Code 2025, Day 6: Trash Compactor

Processing rows and columns of digits.

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


def parse_input(input):
    """
    Returns each row of numbers as a list of lists of integers,
    and returns the signs as a list.
    """
    rows = [[int(x) for x in line.split()] for line in input[:-1]]
    signs = input[-1].split()
    return rows, signs


def mult(lst):
    p = 1
    for x in lst:
        p *= x
    return p


def apply(symbol, values):
    if symbol == '+':
        return sum(values)
    else:
        return mult(values)
    

def compute_column(rows, signs, c):
    """
    Sum/multiply one column of numbers for part 1.
    """
    col = [row[c] for row in rows]
    return apply(signs[c], col)


def part1(rows, signs):
    n_cols = len(rows[0])
    s = sum([compute_column(rows, signs, c) for c in range(n_cols)])
    print(s)


def parse_columns(xp_input, sign_row):
    """
    Return (sign, [values...]) for each column of numbers.
    xp_input is the input transposed, with the sign row removed
    """
    cols = []
            
    c = 0
    while c < len(sign_row):
        if sign_row[c] == ' ':
            c += 1
            continue
        
        c_end = c + 1
        while c_end < len(xp_input) and xp_input[c_end] != '':
            c_end += 1
        values = [int(v) for v in xp_input[c:c_end]]
        cols.append((sign_row[c], values))
        c = c_end + 1

    return cols


def part2(input):
    """
    Transpose the table to make it easier to read the numbers.

    Sample original:
    123 328  51 64
     45 64  387 23
      6 98  215 314
    *   +   *   +

    Transposed:
    1  *
    24
    356
    
    369+
    248
    8
    
     32*
    581
    175
    
    623+
    431
      4

    But strip off the signs and process those separately.
    """
    xp = [line.strip() for line in transpose(input[:-1], ' ')]
    signs_values = parse_columns(xp, input[-1])
    total = sum([apply(*sv) for sv in signs_values])
    print(total)
    

if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()
    rows, signs = parse_input(input)
  
    part1(rows, signs)
    part2(input)
