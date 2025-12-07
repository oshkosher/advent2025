#!/usr/bin/env python3

"""
Advent of Code 2025, Day 7: Laboratories

Splitting and rejoining flows

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


def count_splits(grid, split_points, r, c):
    while r < len(grid) and grid[r][c] == '.':
        r += 1

    if r >= len(grid):
        return

    if grid[r][c] != '.':
        assert grid[r][c] == '^'

    key = r,c
    if key in split_points: return
    split_points.add(key)
    # print(f'split at {r}, {c}')
        
    count_splits(grid, split_points, r, c-1)
    count_splits(grid, split_points, r, c+1)


def part1(grid, start_col):
    split_points = set()
    count_splits(grid, split_points, 1, start_col)
    print(len(split_points))


def part2(grid, start_col):
    height = len(grid)
    width = len(grid[0])

    # track how many paths end up passing through this cell
    count_grid = [[0]*width for _ in range(height)]

    # start with one path below the 'S'
    count_grid[0][start_col] = 1

    for r in range(1, height):
        for c in range(0, width):
            parent_count = count_grid[r-1][c]
            
            # if empty, add my incoming flow
            if grid[r][c] == '.' and parent_count:
                count_grid[r][c] += parent_count

            # if a splitter, add the flow from the cell above me to my left and right
            elif grid[r][c] == '^' and parent_count:
                if c > 0:
                    count_grid[r][c-1] += parent_count
                if c+1 < width:
                    count_grid[r][c+1] += parent_count

        # print(' '.join([str(cell) for cell in count_grid[r]]))

    s = sum(count_grid[height-1])
    print(s)
    


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()

    # read input as a Grid object, where each row is a string
    grid = grid_read(input_filename(), False)
    start_col = grid[0].index('S')
  
    part1(grid, start_col)
    part2(grid, start_col)
