#!/usr/bin/env python

"""
Advent of Code 2025, Day 4: Printing Department

Repeated grid neighbor counting

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
    """
    Add all the nonempty neighbors of (r,c) to the todo set
    """
    for r, c in grid_enumerate_neighbors_8(grid, r, c):
        if grid[r][c] == '@':
            todo.add((r,c))
        

def part2(input):
    todo = set()
    n_removed = 0

    # first pass: test everyone
    # When a cell is removed, add its neighbors to the todo list
    # to be rescanned.
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if is_movable(grid, r, c):
                grid[r][c] = '.'
                n_removed += 1
                q_neighbors(todo, grid, r, c)

    while len(todo) > 0:
        # pull one off the todo list, see if it can be removed
        r, c = todo.pop()
        if is_movable(grid, r, c):
            grid[r][c] = '.'
            n_removed += 1
            q_neighbors(todo, grid, r, c)

    print(n_removed)
        

def part2_dumb(input):
    """
    Don't use todo set, just do full scans until no progress is made.
    This is only 4x slower than using the todo set.
    """
    n_removed = 0
    progress = True

    # do full passes until no progress is made
    while progress:
        progress = False
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if is_movable(grid, r, c):
                    grid[r][c] = '.'
                    n_removed += 1
                    progress = True

    print(n_removed)


if __name__ == '__main__':
    # read input as a Grid object, where each row is a list so cells
    # can be modified
    grid = grid_read(input_filename(), True)

    part1(grid)
    timer = time.perf_counter()
    part2(grid)
    # part2_dumb(grid)
    timer = time.perf_counter() - timer
    # print(f'part2 timer: {timer:.6f}')
