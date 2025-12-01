#!/usr/bin/env python

"""
Advent of Code 2025, 



Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


def part1(input):
    p = 50
    zeros = 0
    for line in input:
        sign = 1 if line[0] == 'R' else -1
        dist = int(line[1:])
        p = (p + sign*dist) % 100
        # print(f'{line} {p}')
        if p == 0: zeros += 1
    print(zeros)


def part2(input):
    p = 50
    zeros = 0
    for line in input:
        sign = 1 if line[0] == 'R' else -1
        dist = int(line[1:])
        for i in range(dist):
            p += sign
            if p == 100: p = 0
            if p == -100: p = 0
            if p == 0: zeros += 1
    print(zeros)


"""
        p = p + sign*dist
        print(f'{line} {p}')
        if p == 0:
            print('  at zero')
            zeros += 1
        else:
            while p >= 100:
                p -= 100
                zeros += 1
                print(f'  -> {p}')
            while p < 0:
                p += 100
                zeros += 1
                print(f'  -> {p}')

"""

if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()
  
    part1(input)
    part2(input)
