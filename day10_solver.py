#!/usr/bin/env python3

import math
from dataclasses import dataclass

from advent import *

from day10_problems import problem_list
# from day10_small_problems import problem_list


"""
Given a list of tuples (A, b, x), where A . x = b, b is the goal,
and x is a solution computed by the 'pulp' linear programming library.
For each tuple, there are multiple solutions x, but the one that minimized
the sum of the values in x is correct. Also, all the values in x must
be non-negative integers.
"""



  
def test_solutions(problem_list):
    for i, (A, b, x) in enumerate(problem_list):
        print(f'problem {i}')
        matrix_print(A)
        
        print(f'soln {x!r}')
    
        prod = matrix_apply(A, x)
        print(f'prod {prod!r}')
        print(f'goal {b!r}')
    
        if prod != b:
            print('  ERROR fail')
    
    
        print()


def max_values(A, jolts):
    """Given the original matrix input and goal joltages,
    return an array of length len(A[0]) containing the maximum possible
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


def count_column_nonzeros(matrix, c):
    n = 0
    for r in range(matrix.height):
        if matrix[r][c] != 0:
            n += 1
    return n


def next_unknown(matrix):
    """
    Given a matrix in row-reduced form, returns the index of a column
    that has nonzero values in multiple rows.
    """
    for c in range(matrix.width-1):
        if count_column_nonzeros(matrix, c) > 1:
            return c
    return -1


def are_all_zeros(lst):
    for e in lst:
        if e != 0:
            return False
    return True


def are_all_non_negative(lst):
    for e in lst:
        if e < 0:
            return False
    return True


def are_all_integers(lst):
    for e in lst:
        if not isinstance(e, int):
            # print(f'non-integer {e!r}')
            return False
    return True


def is_solution_matrix(A):
    n_variables = A.width-1
    if A.height < n_variables:
        # print('  no good--too short')
        return False
    
    for r, row in enumerate(A.rows):
        if r < n_variables:
            if not (are_all_zeros(row[:r])
                    and 1 == row[r]
                    and are_all_zeros(row[r+1:n_variables])):
                # print(f'  no good, row {r} is not a single zero {row!r}')
                return False
        else:
            if not are_all_zeros(row):
                # print(f'  no good, row{r} is no all zeros {row!r}')
                return False

    # print(f'check non-negative {A.column(-1)[:n_variables]}')
    knowns = A.column(-1)[:n_variables]
    if not are_all_non_negative(knowns):
        # print(f'  no good--cannot have negatives')
        return False

    if not are_all_integers(knowns):
        # print(f'  no good--cannot have fractions')
        return False

    return True


@dataclass
class RecurseData:
    original: Matrix
    goal_vector: list
    soln_max_values: list
    best_sum: int
    best_soln: list


def solve_recurse(d, A, depth = ''):
    """
    Recursively solve the matrix A.
    Each time an underconstrained column is found, try every value from
    0 to soln_max_values[column_idx] (inclusive) and recurse.

    Return the solution vector with the lowest sum of values.

    d = RecurseData object that holds my persistent data
    """

    VERBOSE = 0

    if VERBOSE > 1:
        print(f'solve_recurse depth {len(depth)}')

    A.row_reduce()
    if VERBOSE > 2:
        print('reduced')
        A.print()
        print()

    guess_c = next_unknown(A)

    if guess_c == -1:
        # solution found?
        if not is_solution_matrix(A):
            if VERBOSE > 1:
                print('Out of unknowns, but not a solution matrix')
                # A.print()
                # print()
            return

        soln = A.column(-1)[:A.width-1]
        soln_sum = sum(soln)

        if VERBOSE:
            print(f'{depth}solution with sum {soln_sum} found: {soln!r}')

        if soln_sum < d.best_sum:
            if VERBOSE:
                print(f'{depth}a new best')
            d.best_soln = soln
            d.best_sum = soln_sum

        return

    if VERBOSE > 1:
        print(f'{guess_c=}')

    # remove junk zero rows from the end
    n_zero_rows = A.count_tail_zero_rows()
    A.remove_rows(-n_zero_rows, n_zero_rows)

    # add a row that we'll use to test values
    A.append_row([0] * A.width)
    A[-1][guess_c] = 1

    depth = depth + ' '

    # for value in range(0, d.soln_max_values[guess_c]+1):
    
    for value in range(13, d.soln_max_values[guess_c]+1):
        # make a copy of A to mess with
        A_copy = A.copy()
        A_copy[-1][-1] = value

        if VERBOSE:
            print(f'{depth}Column {guess_c}, try {value}')
            if VERBOSE > 2:
                A_copy.print()
                print()

        solve_recurse(d, A_copy, depth)
    

def solve(original_matrix, b):
    # original_matrix.print()
    # print()
    
    assert len(b) == original_matrix.height

    soln_max_values = max_values(original_matrix, b)
    # print(f'max value for each entry: {soln_max_values!r}')

    # augment the matrix with the goal values
    augmented = original_matrix.copy()
    augmented.append_col(b)

    recurse_data = RecurseData(original_matrix, b, soln_max_values, 2**31, [])

    solve_recurse(recurse_data, augmented)
    
    print(f'best found:    {recurse_data.best_soln!r}')


def find_solutions(problem_list):
    for i, (A, b, x) in enumerate(problem_list):
        matrix = Matrix(A)
        print(f'problem {i}')
        print(f'known solution {x!r}')
        solve(matrix, b)
        print()


# test_solutions(problem_list)

find_solutions(problem_list)

# matrix = [
#     [5, -1, 2, 34],
#     [0, 4, -3, 12],
#     [10, -2, 1, -4],
#     ]
# matrix_row_reduce(matrix)

          
