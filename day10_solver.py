#!/usr/bin/env python3

import math
from dataclasses import dataclass

import signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

from advent import *

from day10_problems import problem_list
# from day10_small_problems import problem_list


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
    # for c in range(matrix.width-1):
    for c in range(matrix.width-2, 0, -1):
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


def is_solution_matrix(A, reasons = None):
    n_variables = A.width-1
    if A.height < n_variables:
        if reasons:
            reasons.append('not enough solved variables')
        # print('  no good--too short')
        return False
    
    for r, row in enumerate(A.rows):
        if r < n_variables:
            if not (are_all_zeros(row[:r])
                    and 1 == row[r]
                    and are_all_zeros(row[r+1:n_variables])):
                # print(f'  no good, row {r} is not a single zero {row!r}')
                reasons.append('value rows wrong form')
                return False
        else:
            if not are_all_zeros(row):
                # print(f'  no good, row{r} is no all zeros {row!r}')
                reasons.append('non-value rows have nonzeros')
                return False

    is_good = True
    knowns = A.column(-1)[:n_variables]
    
    if not are_all_integers(knowns):
        # print(f'  no good--cannot have fractions')
        reasons.append('fractions')
        is_good = False

    # print(f'check non-negative {A.column(-1)[:n_variables]}')
    if not are_all_non_negative(knowns):
        # print(f'  no good--cannot have negatives')
        reasons.append('negatives')
        is_good = False

    return is_good


@dataclass
class RecurseData:
    original: Matrix
    goal_vector: list
    soln_max_values: list
    best_sum: int
    best_soln: list
    unknowns: list


def solve_recurse(d, A, depth = ''):
    """
    Recursively solve the matrix A.
    Each time an underconstrained column is found, try every value from
    0 to soln_max_values[column_idx] (inclusive) and recurse.

    Return the solution vector with the lowest sum of values.

    d = RecurseData object that holds my persistent data
    """

    verbose = 0

    # if verbose > 1:
    #     print(f'solve_recurse depth {len(depth)}')

    A.row_reduce()
    if verbose > 2:
        print('reduced')
        A.print()
        print()

    guess_c = next_unknown(A)
    # print(f'{guess_c=}')

    if guess_c == -1:
        # solution found?

        if verbose > 1:
            print(f'{depth}  unknowns = {" ".join([str(x) for x in d.unknowns])}')
        
        reasons = []
        if not is_solution_matrix(A, reasons):
            if verbose > 1:
                # print(f'{depth}  not a solution: {", ".join(reasons)}')
                if verbose > 2:
                    A.print()
                    print()
            return

        soln = A.column(-1)[:A.width-1]
        soln_sum = sum(soln)

        if verbose:
            print(f'{depth}solution with sum {soln_sum} found: {soln!r}')

        if soln_sum < d.best_sum:
            if verbose:
                print(f'{depth}a new best')
            d.best_soln = soln
            d.best_sum = soln_sum

        return

    # if verbose > 1:
    #     print(f'{guess_c=}')

    # remove junk zero rows from the end
    n_zero_rows = A.count_tail_zero_rows()
    A.remove_rows(-n_zero_rows, n_zero_rows)

    # add a row that we'll use to test values
    A.append_row([0] * A.width)
    A[-1][guess_c] = 1

    depth = depth + ' '

    # for value in range(0, d.soln_max_values[guess_c]+1):

    d.unknowns.append(0)
    
    for value in range(0, d.soln_max_values[guess_c]+1):
        d.unknowns[-1] = value
        
        # make a copy of A to mess with
        A_copy = A.copy()
        A_copy[-1][-1] = value

        if verbose:
            # print(f'{depth}Column {guess_c}, try {value}')
            if verbose > 2:
                A_copy.print()
                print()

        solve_recurse(d, A_copy, depth)
    
    del d.unknowns[-1]


def matrix_paste(dest, src):
    assert dest.height == src.height and dest.width == src.width
    for r in range(dest.height):
        dest[r][:] = src[r]


def write_nonfree_formulas(A, free_cols):
    """
    Given a matrix in rref, for each row with a pivot, compute a formula
    """
    for row in A:
        pass


class ApplyFree:
    """
    Given a matrix in rref and a list of the free columns, prepare
    an object that can quickly compute the values of all variables
    given the values of the free variables.

    For example, if row 0 is: 1 0 0 2 3 5 and free_cols = [3,4]
    then b[0] = 5 - (2 * v[3] + 3 * v[4])
    
    """
    
    def __init__(self, rref_mx, free_cols):
        """
        free_cols list of indices of the free columns
        """
        # colum index, constant, coefficients of each free column
        self.cols = []

    def apply(self, v):
        """
        The value of each free column has been set already in v.
        Use those to set the rest.
        """
        for i, const, coeffs in self.cols:
            x = const
            for j, coeff in coeffs:
                x -= coeff * v[j]
            v[i] = x


def apply_free_cols(A, computed_cols, free_cols, v):
    """
    A is the augmented, rref matrix
    
    computed_cols is a list of column indices that are not free
    free_cols is a list of the free columns.
    
    v is a list of column values where the values of the free columns
    have been filled in.

    This uses the values of the free columns to fill in the rest of v.
    """
    # print('apply_free_cols')
    for r, c in enumerate(computed_cols):
        # print(f'row {r}')
        row = A[r]
        value = row[-1]
        for fc in free_cols:
            value -= row[fc] * v[fc]
            
        # reduce to integer if possible
        if isinstance(value, Fraction) and value.denominator == 1:
            value = value.numerator

        # print(f'  write v[{c}] = {value}')
        v[c] = value
    


def solve_non_recursive(original_matrix, A, goal_vector,
                        soln_max_values):

    A.row_reduce()
    free_cols = find_free_var_columns(A)
    n_free = len(free_cols)
    free_col_max_values = [soln_max_values[f] for f in free_cols]
    computed_cols = [x for x in range(A.width-1) if x not in free_cols]

    print(f'after reduction, free cols = {free_cols!r}, maxes {free_col_max_values!r}, computed cols = {computed_cols!r}')
    # A.print()

    """
    # make sure there is a zero row for each free column
    n_zero_rows = A.count_tail_zero_rows()

    if n_zero_rows < len(free_cols):
        print(f'append {n_free - n_zero_rows} zero rows')
        # if there are not enough zero rows, add some
        for _ in range(n_free - n_zero_rows):
            A.append_row()
        A.print()
    elif n_zero_rows > len(free_cols):
        # if there are too many, remove the excess
        print(f'remove {n_zero_rows - n_free} zero rows')
        excess = n_zero_rows - len(free_cols)
        A.remove_rows(a.height - excess, excess)
        A.print()

    # put a 1 in a free row for each free column
    first_free_row = A.height - n_free
    for i, c in enumerate(free_cols):
        A[first_free_row+i][c] = 1

    print('mark each free column')
    A.print()
    
    B = Matrix(height = A.height, width = A.width)

    free_values = [13, 27, 19]
    matrix_paste(B, A)
    for i in range(n_free):
        B[first_free_row+i][-1] = free_values[i]

    print('substituted')
    B.print()

    B.row_reduce()
    print('reduced')
    B.print()
    """

    best_soln = None
    best_sum = math.inf

    v = [0] * (A.width-1)
    first_free_idx = free_cols[0]
    end = free_col_max_values[0] + 1

    def inc(v):
        i = len(free_cols)-1
        while i >= 0:
            vi = free_cols[i]
            v[vi] += 1
            if v[vi] <= free_col_max_values[i]:
                return True
            v[vi] = 0
            i -= 1
        return False
            
    while v[first_free_idx] < end:
        apply_free_cols(A, computed_cols, free_cols, v)
        # print(repr(v))
        if are_all_integers(v) and are_all_non_negative(v):
            s = sum(v)
            # print(f'  is solution size {s}')
            if s < best_sum:
                print(f'  best so far sum={s} {v!r}')
                best_soln = v[:]
                best_sum = s

        if not inc(v):
            break

    return best_soln
    
    """
    v = [0] * (A.width-1)
    for i, c in enumerate(free_cols):
        v[c] = free_values[i]

    print(f'before apply v={v!r}')
    apply_free_cols(A, computed_cols, free_cols, v)
    print(f'after apply v={v!r}')

    return v
    """
    

def solve(original_matrix, b, problem_idx, known_soln):
    # original_matrix.print()
    # print()
    
    assert len(b) == original_matrix.height

    soln_max_values = max_values(original_matrix, b)
    # print(f'max value for each entry: {soln_max_values!r}')

    # augment the matrix with the goal values
    augmented = original_matrix.copy()
    augmented.append_col(b)

    # augmented.row_reduce()
    # print('initial reduction')
    # augmented.print()
    # print()
    
    # free_cols = find_free_var_columns(augmented)
    # print(f'{len(free_cols)} free columns, problem {problem_idx}: {free_cols!r}')
    # print()
    # return

    # recurse_data = RecurseData(original_matrix, b, soln_max_values, 2**31, [], [])
    # solve_recurse(recurse_data, augmented)
    # soln = recurse_data.best_soln

    soln = solve_non_recursive(original_matrix, augmented, b, soln_max_values)
    print(repr(soln))

    soln_size = sum(soln)
    
    # print(f'best found:    {recurse_data.best_soln!r} (size={soln_size})')

    if soln_size != sum(known_soln):
        print(f'  ERROR  known soln = {sum(known_soln)}, I found {soln_size}')


def reorder_columns(original, b):
    soln_max_values = max_values(original, b)
    
    mapping = [(soln_max_values[i], i) for i in range(original.width)]
    print(repr(mapping))
    mapping.sort(reverse=True)
    print(repr(mapping))

    # make a new matrix where column c was column column_order[c] in original
    column_order = [tup[1] for tup in mapping]
    new_b = [tup[0] for tup in mapping]

    A = Matrix(height = original.height, width = original.width)
    print(f'size {A.height} x {A.width}')
    for r in range(A.height):
        for c in range(A.width):
            A[r][c] = original[r][column_order[c]]
    print('with reordered columns')
    A.print()
    print('\nreduce')
    A.row_reduce()
    A.print()
    


def find_solutions(problem_list):
    skip_list = frozenset([23, 70, 87, 105, 108, 133, 137, 138])
    # for i, (A, b, x) in enumerate(problem_list[6:7]):
    for i, (A, b, x) in enumerate(problem_list[23:24]):
        # if i in skip_list: continue
        matrix = Matrix(A)
        print(f'problem {i}')
        # print(f'known solution {x!r} (size={sum(x)})')
        solve(matrix, b, i, x)

        # reorder_columns(matrix, b)


# test_solutions(problem_list)

find_solutions(problem_list)

# matrix = [
#     [5, -1, 2, 34],
#     [0, 4, -3, 12],
#     [10, -2, 1, -4],
#     ]
# matrix_row_reduce(matrix)

          
