#!/usr/bin/env python

"""
Advent of Code 2025, Day 4: Printing Department



Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


def is_movable(grid, r, c):
    return (grid[r][c] == '@' and
            grid_count_set_neighbors(grid, r, c, '@') < 4)


def part1(grid):
    n_movable = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == '.': continue
            if is_movable(grid, r, c):
                n_movable += 1
    print(n_movable)


def q_neighbors(todo, grid, r, c):
    height = len(grid)
    width = len(grid[0])
    
    if r > 0:
        if c > 0: todo.add((r-1,c-1))
        todo.add((r-1,c))
        if c+1 < width: todo.add((r-1,c+1))
            
    if c > 0: todo.add((r,c-1))
    if c+1 < width: todo.add((r,c+1))
    
    if r+1 < height:
        if c > 0: todo.add((r+1,c-1))
        todo.add((r+1,c))
        if c+1 < width: todo.add((r+1,c+1))
        

def part2(input):
    todo = set()

    n_removed = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if is_movable(grid, r, c):
                grid[r][c] = '.'
                n_removed += 1
                q_neighbors(todo, grid, r, c)

    while len(todo) > 0:
        r, c = todo.pop()
        if is_movable(grid, r, c):
            grid[r][c] = '.'
            n_removed += 1
            if n_removed % 1000 == 0:
                print(f'...{n_removed}')
            q_neighbors(todo, grid, r, c)

    print(n_removed)
    


if __name__ == '__main__':
    # read input as a list of strings
    # input = read_problem_input()

    # read input as a Grid object, where each 
    grid = grid_read(input_filename(), True)

    # part1(grid)
    part2(grid)
