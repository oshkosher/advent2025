#!/usr/bin/env python3

import math
from dataclasses import dataclass
from math import floor, ceil

import signal
# signal.signal(signal.SIGPIPE, signal.SIG_DFL)

from advent import *

from day10_problems import problem_list


"""
Given a list of tuples (A, b, x), where A . x = b, b is the goal,
and x is a solution computed by the 'pulp' linear programming library.
For each tuple, there are multiple solutions x, but the one that minimized
the sum of the values in x is correct. Also, all the values in x must
be non-negative integers.

hardest problems:
3 free columns, problem 23: [10, 11, 12]
3 free columns, problem 70: [7, 8, 9]
3 free columns, problem 87: [9, 10, 11]
3 free columns, problem 105: [10, 11, 12]
3 free columns, problem 108: [9, 11, 12]
3 free columns, problem 133: [4, 6, 7]
3 free columns, problem 137: [10, 11, 12]
3 free columns, problem 138: [10, 11, 12]

https://typst.app/project/puQShEWsytudqS9E9dHtsj

Slow one with 2 free variables, problem 46
"""

def max_values(A, jolts):
    """
    Given the original matrix input and goal joltages,
    returns an array of length len(A[0]) containing the maximum possible
    value of each solution entry.

    This is computed by taking the minimum of at all the entries in
    jolts to which column c contributes. Since the largest factor in
    the matrix is 1 and all solution vector entries are non-negative,
    if an entry in the solution vector is larger than this value it
    will cause the goal joltage to be too high.
    """
    maxes = [math.inf] * len(A[0])

    for r, row in enumerate(A):
        for c, e in enumerate(row):
            if e and jolts[r] < maxes[c]:
                maxes[c] = jolts[r]
    return maxes


def are_all_non_negative(lst):
    for e in lst:
        if e < 0:
            return False
    return True


def are_all_integers(lst):
    for e in lst:
        if not is_integer(e):
            return False
    return True


def find_free_var_columns(matrix):
    """
    Given an augmented matrix in row-reduced eschelon form, where
    each pivot column has a leading '1'.
    """
    without_pivot = set(range(matrix.width-1))
    for r, row in enumerate(matrix):
        for c, e in enumerate(row[:-1]):
            if e != 0:
                without_pivot.remove(c)
                break
    return list(without_pivot)


def is_integer(x):
    return (isinstance(x, int) or
            (isinstance(x, Fraction) and x.denominator == 1))

            
def fract_to_int(v):
    if isinstance(v, Fraction) and v.denominator == 1:
        return v.numerator
    else:
        return v


def apply_fract_to_int(vec):
    for i in range(len(vec)):
        if isinstance(vec[i], Fraction) and vec[i].denominator == 1:
            vec[i] = vec[i].numerator


def compute_solution_vectors(A, free_cols):
    """
    A: RREF augmented matrix
    free_cols: list of free columns

    Returns a 2-tuple:
      0: base vector of length A.width-1
      1: list of (len(free_cols)+1) free vectors, each of length A.width-1

    All solution vectors are linear combinations of base and the free vectors.
    For example: result[0] + 2*result[1][0] + 5*result[1][1]
    """

    # length of each vector == length of solution vector
    n = A.width - 1

    n_empty_rows = A.count_tail_zero_rows()
    n_non_empty_rows = A.height - n_empty_rows

    base = A.column(-1)[:n_non_empty_rows]
    for c in free_cols:
        base[c:c] = [0]

    # print(f'base_vec={base!r}')

    if len(base) != n:
        print(f'bad base len, expected {n}, got {len(base)}')
        A.print()
    
    assert len(base) == n

    free_vecs = []
    for c in free_cols:
        assert A.height <= A.width-1
        v = [0] * n

        read_r = 0
        for r in range(n):
            if r in free_cols:
                if r == c:
                    v[r] = 1
                else:
                    v[r] = 0
            else:
                v[r] = -A[read_r][c]
                read_r += 1
        
        free_vecs.append(v)

        # print(f'free_vec[{c}] = {v!r}')

    return base, free_vecs


def add_scaled_vector(dest, src, v, factor = 1):
    for i in range(len(dest)):
        dest[i] = src[i] + v[i] * factor


def trim_search_range(v, dv, lo, hi):
    """
    v: a base vector
    dv: a difference vector
    lo: minimum factor to check
    hi: maximum factor to check

    Figure out what range of k (bounded by [lo:hi]) yields
    non-negative values for all entries in (v + k dv).

    Also we want to find the solution with the minimum sum of entries
    in (v + k dv), so this uses the sum of values in dv to determine
    whether to search with increasing k or decreasing k, such that the
    first solution found will be the best.

    Result is a range object that defines the search range and direction.
    """

    verbose = False

    for i in range(len(v)):
        if v[i] >= 0:
            if dv[i] < 0:
                limit = floor(v[i] / -dv[i])
                hi = min(limit, hi)
                if verbose:
                    print(f'column {i} stop at {limit}: {lo}..{hi}')
        else:
            if dv[i] <= 0:
                if verbose:
                    print(f'column {i} no-go x={v[i]} dx={dv[i]}')
                hi = -1
            else:
                limit = ceil(v[i] / -dv[i])
                lo = max(limit, lo)
                if verbose:
                    print(f'column {i} start at {limit}: {lo}..{hi}')

        if lo > hi:
            if verbose:
                print('stop now')
            return range(0, 0, 1)

    assert lo <= hi
    
    if sum(dv) < 0:
        if verbose:
            print(f'test {hi} down to {lo}')
        return range(hi, lo-1, -1)
    else:
        if verbose:
            print(f'test {lo} up to {hi}')
        return range(lo, hi+1, +1)


def vec_add_mult_sum_ints(v, dv, k):
    """
    Checks if all entries in (v + k dv) are integers.
    If so, return their sum.
    Otherwise return None.
    """
    s = 0
    for i in range(len(v)):
        x = v[i] + k * dv[i]
        # print(f'  v[{i}] = {x}')
        if not is_integer(x):
            return None
        s += x

    assert is_integer(s)
    return fract_to_int(s)
        

def solve_free1(A, free_col, max_value, base_vec, free_vec):
    """
    Solve a system with one free variable.
    A: row-reduced augmented matrix.
    free_col: index of the free column
    base_vec: null space base vector
    free_vec: null space free vector. The null space is all vectors of the
      form base_vec + k * free_vec

    Depending on whether the sum of free_vec is positive or negative,
    count up from 0 or down from max_value. Return the first solution that
    is all non-negative and integers.
    """
    
    search_range = trim_search_range(base_vec, free_vec, 0, max_value)
    for k in search_range:
        s = vec_add_mult_sum_ints(base_vec, free_vec, k)
        if s:
            return s
        
    return 0


def solve_free2_visual(A, free_cols, max_values,
                       base_vec, free_vecs):
    """
    Draw a 2-d grid of solutions and non-solutions.
    """
    v = [0] * len(base_vec)
    w = [0] * len(base_vec)

    best_r = -1
    best_c = -1
    best_sum = math.inf
    best_soln = [0]

    grid = grid_create(max_values[0]+1, max_values[1]+1, True)
    
    for r in range(max_values[0]+1):
        add_scaled_vector(w, base_vec, free_vecs[0], r)
        for c in range(max_values[1]+1):
            add_scaled_vector(v, w, free_vecs[1], c)
            apply_fract_to_int(v)
            
            all_ints = are_all_integers(v)
            all_pos = are_all_non_negative(v)

            if all_ints:
                if all_pos:
                    grid[r][c] = '#'
                    s = sum(v)
                    if s < best_sum:
                        best_r = r
                        best_c = c
                        best_sum = s
                        best_soln = v[:]
                else:
                    grid[r][c] = '-'
            else:
                grid[r][c] = '/'

    grid[best_r][best_c] = '@'
    grid_print(grid)
    
    return best_soln


def solve_free2(A, max_values, base_vec, free_vecs):

    v = [0] * len(base_vec)
    w = [0] * len(base_vec)

    best_sum = math.inf

    for r in range(max_values[0]+1):
        add_scaled_vector(w, base_vec, free_vecs[0], r)
        search_range = trim_search_range(w, free_vecs[1], 0, max_values[1])
        
        for c in search_range:
            s = vec_add_mult_sum_ints(w, free_vecs[1], c)
            if s is not None:
                if s < best_sum:
                    best_sum = s
                break
                
    return best_sum


def solve_free3(problem_idx, A, max_values, base_vec, free_vecs):
    # print(f'problem {problem_idx}, {max_values=}')

    assert len(free_vecs) == 3 and len(max_values) == 3
    v0 = [0] * len(base_vec)
    v1 = [0] * len(base_vec)

    best_sum = math.inf

    for k0 in range(max_values[0]+1):
        add_scaled_vector(v0, base_vec, free_vecs[0], k0)
        for k1 in range(max_values[1]+1):
            add_scaled_vector(v1, v0, free_vecs[1], k1)
            
            search_range = trim_search_range(v1, free_vecs[2], 0, max_values[2])

            for k2 in search_range:
                s = vec_add_mult_sum_ints(v1, free_vecs[2], k2)
                if s:
                    if s < best_sum:
                        best_sum = s
                    break
        
    return best_sum
    

def solve(original_matrix, b, problem_idx, known_soln, total_elapsed_ns):

    timer_ns = time.perf_counter_ns()
    
    assert len(b) == original_matrix.height

    n_vars = original_matrix.width
    soln_max_values = max_values(original_matrix, b)
    assert n_vars == len(soln_max_values)

    # augment the matrix with the goal values
    A = original_matrix.copy()
    A.append_col(b)
    A.row_reduce()

    # remove empty rows at the bottom
    n_zero_rows = A.count_tail_zero_rows()
    A.remove_rows(A.height - n_zero_rows, n_zero_rows)
    
    free_cols = find_free_var_columns(A)
    n_free = len(free_cols)
    
    base_vec, free_vecs = compute_solution_vectors(A, free_cols)
        
    assert len(free_vecs) == n_free

    free_maxes = [soln_max_values[free_cols[i]] for i in range(n_free)]
    
    if n_free == 0:
        soln = A.column(-1)[:n_vars]
        soln_size = sum(soln)
    elif n_free == 1:
        soln_size = solve_free1(A, free_cols[0], free_maxes[0],
                           base_vec, free_vecs[0])
    elif n_free == 2:
        soln_size = solve_free2(A, free_maxes, base_vec, free_vecs)
    else:
        soln_size = solve_free3(problem_idx, A, free_maxes, base_vec, free_vecs)
                                   
    timer_ns = time.perf_counter_ns() - timer_ns
    # print(f'{problem_idx}\t{timer_ns / 1e9:.6f}s\t{n_free}\t{soln!r}')
    print(f'{problem_idx}\t{timer_ns / 1e9:.6f}s\t{n_free}\t{soln_size}')
    total_elapsed_ns[0] += timer_ns

    # soln_size = sum(soln)
    
    # print(f'best found:    {recurse_data.best_soln!r} (size={soln_size})')

    if known_soln:
        if soln_size != sum(known_soln):
            print(f'  ERROR  known soln = {sum(known_soln)}, I found {soln_size}')
            
    return soln_size


def find_solutions(problem_list):
    button_sum = 0
    total_elapsed_ns = [0]
    for i, (A, b, x) in enumerate(problem_list):
        matrix = Matrix(A)
        button_sum += solve(matrix, b, i, x, total_elapsed_ns)

    print(f'total solve time: {total_elapsed_ns[0] / 1e9:.6f} sec')
    print(button_sum)


if __name__ == '__main__':
    find_solutions(problem_list)
