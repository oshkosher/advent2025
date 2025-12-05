#!/usr/bin/env python3

"""
Advent of Code 2025, Day 5: Cafeteria

Integer ranges

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools
from dataclasses import dataclass

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


@dataclass
class Range:
  """
  yes, this could easily be a 2-element list, but there could be confusion
  on whether it was start/end or start/length, and whether end was inclusive
  """
  start: int
  end: int  # inclusive

  def contains(self, i):
    return self.start <= i <= self.end

  def __lt__(self, other):
    return self.start < other.start

  def __len__(self):
    return self.end - self.start + 1


def is_in_range(range_list, value):
  """
  If we know range_list is sorted and nonoverlapping (which it is after
  calling minimize_ranges), this could be make more efficient by starting
  with a binary search.
  """
  for range in range_list:
    if value < range.start: break
    if range.contains(value):
      return True

  return False


def print_ranges(range_list):
  for r in range_list:
    print(f'  {r}, len = {len(r)}')


def minimize_ranges(range_list):
  range_list.sort()

  # print('after sort, before minimize')
  # print_ranges(range_list)
  
  i = 1
  while i < len(range_list):
    prev = range_list[i-1]
    r = range_list[i]
    assert prev.start <= r.start

    # print(f'evaluate {i=} {prev=} {r=}')

    # no overlap
    if r.start > prev.end+1:
      # print('  no overlap')
      i += 1
      continue

    # overlap
    prev.end = max(prev.end, r.end)
    del range_list[i]
    # print(f'  overlap, new end = {prev.end}')
    # print_ranges(range_list)


def total_range_list_coverage(range_list):
  return sum([len(r) for r in range_list])


def parse_input(input):
  range_list = []

  # before the first blank line the input contains ranges like 3-10
  i = 0
  while len(input[i]) > 0:
    s, e = [int(x) for x in input[i].split('-')]
    range_list.append(Range(s, e))
    i += 1

  i += 1

  # after the blank line each line is one of the sample values
  ingrs = []
  while i < len(input):
    ingrs.append(int(input[i]))
    i += 1

  # sort the ranges and remove overlaps
  minimize_ranges(range_list)
  # print(repr(range_list))

  return range_list, ingrs
    


def part1(range_list, ingrs):

    n_fresh = 0
    for ingr in ingrs:
      if is_in_range(range_list, ingr):
        # print(f'{ingr} is ok')
        n_fresh += 1

    print(n_fresh)


def part2(range_list):
    print(total_range_list_coverage(range_list))


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()
    range_list, ingrs = parse_input(input)
    
    part1(range_list, ingrs)
    part2(range_list)
