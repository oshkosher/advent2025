#!/usr/bin/env python

"""
Advent of Code 2025, Day 2: Gift Shop

Looking for numbers that consist of repeated digits.

Version 1: use strings
Version 2: no strings; do all checks with integer math

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


def is_twice_ints(i):
    ilen = int_len(i)
    if ilen & 1: return False   # can't work with odd number of digits
    # half_len = ilen // 2
    mask = 10 ** (ilen // 2)
    return (i % mask) == (i // mask)


def all_dups_match(s, prefix_len):
    # use the 'skip' argument for range
    # for example, if len(s)==15 and prefix_len==5, this will yield [5, 10]
    # Check the prefix against each of these substrings.
    for p in range(prefix_len, len(s), prefix_len):
        if s[:prefix_len] != s[p:p+prefix_len]:
            return False
    return True


def is_all_repeats(i):
    s = str(i)
    slen = len(s)
    # print(f'test {i}, {slen=}')

    # test every length of duplicated substring
    for prefix_len in range(1, slen//2+1):
        
        # skip lengths that don't divide the string length equally
        if slen % prefix_len != 0: continue

        # check that the rest of the string is all duplicates of this prefix
        if all_dups_match(s, prefix_len):
            return True
        
    return False


def all_dups_match_ints(i, mask):
    """
    mask is a power of 10
    For example, to check a 3-digit prefix, mask will be 1000
    Use this both to strip off the lower digits, and to shift down
    i after each successful match.
    """
    prefix = i % mask
    i //= mask

    while i > 0:
        if i % mask != prefix: return False
        i //= mask

    return True


def is_all_repeats_ints(i):
    mask = 10
    mask_len = 1

    ilen = int_len(i)
    half_len = ilen//2
    while mask_len <= half_len:
        if ilen % mask_len == 0:
            if all_dups_match_ints(i, mask):
                return True

        mask *= 10
        mask_len += 1

    return False


def sum_invalids(ranges, invalid_fn):
    sum_invalid = 0
    
    for begin, end in ranges:
        # print(f'{begin}-{end}')
        for i in range(begin, end+1):
            if invalid_fn(i):
                sum_invalid += i
                # print(f'  {i}')

    print(sum_invalid)


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()
    ranges = [[int(x) for x in r.split('-')] for r in input[0].split(',')]

    sum_invalids(ranges, is_twice_ints)
    sum_invalids(ranges, is_all_repeats_ints)
    
