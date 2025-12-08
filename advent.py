#!/usr/bin/env python3

"""
Common code for Advent of Code puzzles.

Ed Karrels, ed.karrels@gmail.com, December 2025

input_filename() - figure out name of input file based on sys.argv[1]
read_problem_input(filename=None) - read input into list of strings, no newlines
read_grid(inf, lists=False) - read list of strings or lists
create_grid(rows, cols, lists, fill='.') - create empty grid
grid_get(grid, (row,col)) - use tuple to read grid cell
paste_grid(dest_grid, dest_row, dest_col, src_grid)
  copy one grid onto another
grid_add_border(grid, border_width=1, fill='.') - return padded grid
grid_row_to_string(row) - convert row to string
print_grid(grid) - print grid to stdout
grid_to_string(grid) - make grid into one big string
grid_search(grid, target) - return (r,c) where grid[r][c]==target
grid_count(grid, target) - count number of copies of target
grid_nb8(grid, r, c) - iterable 8 neighbors, with range check
grid_count_set_neighbors(grid, r, c, value = '#')
  count neighbors (including diagonals) with this value
grid_deep_copy(grid) - make a copy
sparse_read(file or string[], empty='.') - return dict {(r,c): 'x'}
sparse_size(sparse_grid) - returns (min_row, height, min_col, width)
sparse_to_grid(sparse_grid, empty='.', lists=False) - convert to dense grid
movep(pos, d, count=1) - move 2-list pos [r,c] in direction d
  direction 0=North, 1=East, 2=South, 3=West
move(r, c, d, count=1) - return (r,c) moved in direction d
dir_inv(d) - given a direction, return the opposite direction
right_turn(d) - return direction 90 to the right
left_turn(d) - 90 left
turn_angle(prev, dir) - compute direction difference, -90, 0, 90, or 180
list2str(lst) - make list into string, calling str(), ' ' in between
list_sum(lst) - add up a list
list_prod(lst) - product of all list values
is_ordered(lst) - return True iff lst is ordered
column(grid,c) - return column c of the grid, transposed
transpose(grid) - return transposed grid
sliding_window(sequence, window_size = 2) - generates sliding window tuples
clump(sequence, size=2) - generate chunks (0,1), (2,3), (4,None)
prime_factorization(x) - [(factor1,power1), (factor2,power2), ...]
prod_powers(factor_list) - invert prime_factorization()
sum_factors(x) - sum of all factors of x. i.e. 12=>28
chinese_remainder_theorem([(m, a), ...]) - return t s.t. t%m==a for all
freq(sequence) - returns ordered frequency list: [('e', 37), ..., ('q', 1)]
ints(string) - return a list of integers in the string
int_len(n) - returns len(str(n))
count_set_bits(integer) - returns the number of '1' bits in an integer
matrix_solve(matrix) - Gram-Schmidt matrix reduction
matrix_print(matrix)
dijkstra(container, origin_node) - applies Dijkstra's algorithm
md5(s) - returns lowercase base-16 MD5 hash of ASCII string s
"""

import re, sys, collections, time, math, hashlib
from fractions import Fraction
from primes import prime_list
from typing import Optional
import eheap
# import draw_grid

UP = NORTH = 0
RIGHT = EAST = 1
DOWN = SOUTH = 2
LEFT = WEST = 3

direction_vector = ((-1,0), (0,1), (1,0), (0,-1))

compass_names = ('NORTH', 'EAST', 'SOUTH', 'WEST')
direction_names = ('UP', 'RIGHT', 'DOWN', 'LEFT')
direction_letters = ('U', 'R', 'D', 'L')
letter_to_direction = {letter: i for i, letter in enumerate(direction_letters)}
direction_chars = ('^', '>', 'v', '<')
char_to_direction = {ch: i for i, ch in enumerate(direction_chars)}

script_re = re.compile(r'day(\d+)\.py')

def input_filename():
    """
    If a command line argument was provided, use that.
    Otherwise, guess what the name of the input file is based on the name
    of the script.
    The script is expected to have a name in the form day<n>.py.
    If none can be guessed, throw an exception.
    """
    if len(sys.argv) > 1:
        return sys.argv[1]

    match = re.compile(r'day(\d+)\.py').search(sys.argv[0])
    if match:
        return f'day{match.group(1)}.in'

    raise Exception(f'advent.input_filename() failed with sys.argv[0] = {sys.argv[0]!r}')


def read_problem_input(filename = None):
    """
    Use input_filename() to get the name of the input file, read the file,
    and strip off newlines with rstrip().
    warning: this will remove all trailing whitespace, so don't use this
    if trailing whitespace is significant for the problem.
    Returns a list of strings.
    This will throw an exception if the file cannot be found.
    """
    if filename == None:
        filename = input_filename()
    with open(filename) as inf:
        lines = inf.readlines()

    for i, line in enumerate(lines):
        lines[i] = line.rstrip()
    return lines


class Grid:
    """
    2-D grid stored as a list of lists.
    Vertical coordinate commes first: grid[row][col]
    """
    def __init__(self, height, width, fill_value = '.'):
        self.height = height
        self.width = width
        self.rows = [[fill_value for _ in range(width)] for _ in range(height)]

    @staticmethod
    def read(inf):
        """
        Reads a grid from a file.
        Returns the grid.
        inf can be a file object or a string filename.
        """
        filename = None
        if isinstance(inf, str):
            filename = inf
            inf = open(filename)

        grid = Grid(0, 0)
        grid.width = None
        
        while True:
            line = inf.readline().rstrip()
            if line == '': break
            
            if grid.width == None:
                grid.width = len(line)
            else:
                if len(line) != grid.width:
                    raise Exception('Error reading grid. '
                                    'Inconsistent row lengths')
            grid.rows.append(list(line))

        grid.height = len(grid.rows)
            
        if filename:
            inf.close()

    def __getitem__(self, i):
        return self.rows[i]

    def get(self, pos):
        row, col = pos
        return self.rows[row][col]

    def set(self, pos, value):
        row, col = pos
        self.rows[row][col] = value

    def row_to_string(self, r):
        return ''.join([str(e) for e in self.rows[r]])

    def print(self):
        for r in range(self.height):
            print(self.row_to_string(r))

    def search(self, value):
        """
        Returns an iterable of (row,col) tuples for every cell
        containing value.
        """
        for r in range(self.height):
            row = self.rows[r]
            for c in range(self.width):
                if row[c] == value:
                    yield r, c

    def count(self, value):
        """
        Returns the number of cells containing the given value.
        """
        k = 0
        for r in range(self.height):
            row = self.rows[r]
            for c in range(self.width):
                if row[c] == value:
                    k += 1
        return k

    def nb8(self, r, c):
        """
        Returns iterable of (r,c) coordinates of all neighbors which
        are within bounds of the grid. This includes diagonals such
        as (r-1, c+1).
        """
        
        not_left = (c > 0)
        not_right = (c+1 < self.width)

        if r > 0:
            if not_left: yield r-1, c-1
            yield r-1, c
            if not_right: yield r-1, c+1

        if not_left: yield r, c-1
        if not_right: yield r, c+1

        if r+1 < self.height:
            if not_left: yield r+1, c-1
            yield r+1, c
            if not_right: yield r+1, c+1


    def nb8_count(self, r, c, value = '#'):
        """
        Returns the number of neighbors (including diagonals) with the
        given value.
        """
        k = 0
        for ri, ci in self.nb8(r, c):
            if self.rows[ri][ci] == value:
                k += 1
        return k


class SparseGridRow:
    def __init__(self, sparse_grid, r):
        self.sparse_grid = sparse_grid
        self.r = r

    def __getitem__(self, c):
        return self.sparse_grid.get((self.r,c))

    def __setitem__(self, c, value):
        self.sparse_grid.set((self.r,c), value)
        

class SparseGrid:
    def __init__(self):
        self.row_min = self.row_max = self.col_min = self.col_max = None
        self.data = {}

    def set(self, pos, value):
        self.data[pos] = value

    def get(self, pos, default='.'):
        return self.data.get(pos, default)

    def __getitem__(self, r):
        return SparseGridRow(self, r)


def grid_read(inf, split_rows_into_lists = False):
    """
    Read a 2-d grid.
    inf can be either a filename or input stream.
    If split_rows_into_lists is True, the return a list of list of characters.
    Otherwise, return a list of strings.

    A list of a list of characters is useful if the cells will be modified,
    because strings are immutable.
    """
    filename = None
    if isinstance(inf, str):
        filename = inf
        inf = open(filename)

    rows = []
    while True:
        line = inf.readline().rstrip()
        if line == '': break
        if split_rows_into_lists:
            line = [c for c in line]
        rows.append(line)

    if filename:
        inf.close()

    return rows


def grid_create(row_count, col_count, split_rows_into_lists=False, fill='.'):
    """
    Create an empty grid with (row_count) rows and (col_count) columns.
    If (split_rows_into_lists) is true, then each row is a list, otherwise
    a string. Fill each cell with (fill).
    """
    if split_rows_into_lists:
        return [[fill]*col_count for _ in range(row_count)]
    else:
        return [fill*col_count for _ in range(row_count)]


def grid_get(grid, coord):
    """
    Use 2-tuple coordinate (row,col) to read a cell from the grid.
    This works if grid is a list of strings or a list of lists.
    """
    return grid[coord[0]][coord[1]]


def grid_paste(dest_grid, dest_row, dest_col, src_grid):
    """
    Copy one grid onto another grid.
    This works if the source and destination grids are lists of strings or
    lists of lists.
    """
    src_width = len(src_grid[0])
    if type(dest_grid[0]) == list:
        if type(src_grid[0]) == list:
            for r, src_row in enumerate(src_grid):
                dest_grid[dest_row+r][dest_col:dest_col+src_width] = src_row
        else:
            for r, src_row in enumerate(src_grid):
                dest_grid[dest_row+r][dest_col:dest_col+src_width] \
                  = [x for x in src_row]
    else:
        if type(src_grid[0]) == str:
            for r, src_row in enumerate(src_grid):
                tmp = dest_grid[dest_row+r]
                dest_grid[dest_row+r] = (tmp[:dest_col] + src_row
                                         + tmp[dest_col+src_width:])
        else:
            for r, src_row in enumerate(src_grid):
                tmp = dest_grid[dest_row+r]
                dest_grid[dest_row+r] = \
                  tmp[:dest_col] + ''.join(src_row) + tmp[dest_col+src_width:]


def grid_add_border(grid, border_width=1, fill='.'):
    """
    Add a border of (border_width) rows and columns around a grid. A new grid is returned.
    """
    is_string = isinstance(grid[0], str)
    padded = grid_create(len(grid) + border_width*2,
                        len(grid[0]) + border_width*2,
                        not is_string,
                        fill)
    grid_paste(padded, border_width, border_width, grid)
    return padded


def grid_row_to_string(row):
    """
    Given a list of element or a string, return a string of the elements with
    no separators between them.
    """
    if isinstance(row, list):
        return ''.join([str(e) for e in row])
    else:
        return row
  

def grid_print(grid):
    """
    Print a grid to stdout.
    """
    for row in grid:
        print(grid_row_to_string(row))

    
def grid_to_string(grid):
    """
    Convert a grid to one big string.
    """
    return '\n'.join([grid_row_to_string(row) for row in grid])
  


def grid_search(grid, target):
    """
    Search grid for an element equal to (target).
    On success, return (row, column) tuple.
    On failure, return None.
    """
    for r, row in enumerate(grid):
        for c, e in enumerate(row):
            if e == target:
                return (r, c)
    return None


def grid_count(grid, target):
    """
    Returns the number of elements in the grid equal to (target).
    """
    n = 0
    for row in grid:
        for e in row:
            if e == target:
                n += 1
    return n


def grid_nb8(grid, r, c):
    """
    Returns iterable of (r,c) coordinates of all neighbors which
    are within bounds of the grid. This includes diagonals such
    as (r-1, c+1).
    """
    height = len(grid)
    width = len(grid[0])
    # assert(0 <= r < height and 0 <= col < width)

    not_left = (c > 0)
    not_right = (c+1 < width)

    if r > 0:
        if not_left: yield r-1, c-1
        yield r-1, c
        if not_right: yield r-1, c+1
      
    if not_left: yield r, c-1
    if not_right: yield r, c+1
    
    if r+1 < height:
        if not_left: yield r+1, c-1
        yield r+1, c
        if not_right: yield r+1, c+1


def grid_count_set_neighbors(grid, r, c, value = '#'):
    """
    Of the 8 neighboring cells from (r,c), this returns the number
    whose value is (value).
    """
    
    count = 0
    for r1, c1 in grid_nb8(grid, r, c):
        if grid[r1][c1] == value:
            count += 1
  
    return count


def grid_deep_copy(grid):
    return [r.copy() for r in grid]


def grid_read_sparse(input, empty='.'):
    """
    Read a grid from a file (or list of strings) and return a dictionary of
    all the cells that are not empty. (row,col): character

    For example, given:
    .#.
    .x.
    S..

    this would return {(0, 1): '#', (1, 1): 'x', (2,0): 'S'}
    """
    sparse = {}
    for r, line in enumerate(input):
        # print(f'read {line.rstrip()!r}')
        line = line.rstrip()
        for c, char in enumerate(line):
            if char != empty:
                sparse[(r,c)] = char
    return sparse


def sparse_size(sparse_grid):
    """
    Returns (min_row, height, min_col, width)
    """
    first = True
    for r,c in sparse_grid.keys():
        if first:
            first = False
            min_r = max_r = r
            min_c = max_c = c
        else:
            min_r = min(min_r, r)
            min_c = min(min_c, c)
            max_r = max(max_r, r)
            max_c = max(max_c, c)
    return (min_r, max_r - min_r + 1,
            min_c, max_c - min_c + 1)


def sparse_to_grid(sparse_grid, empty='.', split_rows_into_lists=False):
    """
    Given a sparse grid, fill in a dense grid.
    """
    (min_row, height, min_col, width) = sparse_size(sparse_grid)

    def makeRow(r):
        row = [sparse_grid.get((r,c), empty)
               for c in range(min_col, min_col+width)]
        if split_rows_into_lists:
            return row
        else:
            return ''.join(row)

    return [makeRow(r) for r in range(min_row, min_row + height)]

        
def test_sparse_read():
    input = ['.#.', '.x.', 'S..']
    s = grid_read_sparse(input)
    print(repr(s))

    print(repr(sparse_size(s)))

    grid_print(sparse_to_grid(s, '-', False))
    
    # input = open('2023/day21.small')
    # sparse_read(input)


def movep(pos, d, count=1):
    """
    pos is a list [row, col] which is modified in-place
    """
    vec = direction_vector[d]
    pos[0] += count * vec[0]
    pos[1] += count * vec[1]


def move(r, c, d, count=1):
    """
    (r, c) is the starting position
    Returns new_r, new_c
    """
    vec = direction_vector[d]
    return r + count * vec[0], c + count * vec[1]

  
def dir_inv(d):
    d += 2
    if d > 3: d -= 4
    return d


def right_turn(d):
    if d == LEFT:
        return UP
    else:
        return d + 1

  
def left_turn(d):
    if d == UP:
        return LEFT
    else:
        return d - 1

  
def turn_angle(p, d):
    # previous_direction, new_direction
    if p == d: return 0
    elif d == right_turn(p): return 90
    elif d == left_turn(p): return -90
    else: return 180


def testPasteGrid():
    src = grid_create(3, 5, True, 'o')
    dest = grid_create(10, 10, False, '.')
    grid_paste(dest, 1, 2, src)
    grid_print(dest)

  
def list2str(lst):
    return ' '.join([str(x) for x in lst])


def list_sum(lst):
    if len(lst) == 0: return 0
    s = lst[0]
    for x in lst[1:]:
        s += x
    return s


def list_prod(lst):
    if len(lst) == 0: return 1
    s = lst[0]
    for x in lst[1:]:
        s *= x
    return s


def is_ordered(lst):
    for i in range(1, len(lst)):
        if lst[i] < lst[i-1]:
            return False
    return True


def column(grid, c, is_str):
    col = [row[c] for row in grid]
    if is_str:
        return ''.join(col)
    else:
        return col
    

def in_range_or_fill(lst, i, fill):
    if i < len(lst):
        return lst[i]
    else:
        return fill


def column_uneven(grid, c, is_str, fill):
    col = [in_range_or_fill(row, c, fill) for row in grid]
    if is_str:
        return ''.join(col)
    else:
        return col


def transpose(rows, fill = None, is_fixed_width = False):
    """
    Returns a transposition of rows.
    If the caller knows that every row in rows has the same length,
    they can set is_fixed_width to True, and some checks will be skipped.

    If rows[0] is a string, results will be strings.
    """
    is_str = type(rows[0]) == str
    
    if is_fixed_width:
        width = len(rows[0])
        return [column(rows, c, is_str) for c in range(width)]
    else:
        width = max([len(line) for line in rows])
        return [column_uneven(rows, c, is_str, fill) for c in range(width)]


def sliding_window(sequence, window_size = 2):
    """
    Generator function returning (window_size)-element sliding windows of
    the input sequence.
    If len(sequence) < window_size, returns nothing.
    (there might be something in the itertools package to do this already)
    """
    for i in range(len(sequence)-(window_size-1)):
        yield sequence[i:i+window_size]


def clump(sequence, size = 2):
    """
    Generator function returning disjoint consecutive groups of sequence.
    If size does not divide length evenly, last sequence is short.
    For example, given ('foobar', 2): ['fo', 'ob', 'ar']
    or ('foobar', 3): ['foo', 'bar']
    or ('foobar', 4): ['foob', 'ar']
    """
    if hasattr(sequence, "__getitem__"):
        start = 0
        end = len(sequence)
        while start < end:
            yield sequence[start:start+size]
            start += size
    else:
        sublist = []
        for x in sequence:
            sublist.append(x)
            if len(sublist) == size:
                yield tuple(sublist)
                sublist.clear()
        if len(sublist) > 0:
            yield tuple(sublist)


def testClump():
    c = list(range(20))
    for i in range(1, 11):
        a = clump(c, 2)
        print(i)
        for x in clump(a, i):
            print(repr(x))
        

def prime_factorization(x):
    """
    Returns [(prime, power), ...] such that the product of prime_i^power_k
    is x.
    """
    assert isinstance(x, int)
    assert x > 0
    factor_list = []
    orig_x = x
    
    n_twos = 0
    while (x&1) == 0:
        x >>= 1
        n_twos += 1
    if n_twos:
        factor_list.append((2, n_twos))

    prime_idx = 1

    # count up to sqrt(x)
    while (p := prime_list[prime_idx])**2 <= x:
        # print(f'try {p}')
        if x % p == 0:
            count = 1
            x //= p
            while x % p == 0:
                count += 1
                x //= p
            factor_list.append((p, count))
        prime_idx += 1

    if x > 1:
        # x must be prime
        factor_list.append((x, 1))

    assert prod_powers(factor_list) == orig_x, \
        f'wrong factorization of {orig_x}: {factor_list!r}'

    # print(f'{orig_x} = {factor_list!r}')
        
    return factor_list


def prod_powers(factor_list):
    """
    Invert prime_factorization(x).
    Given a list of [(number, power), ...], compute the product
    of all number_k**power_k.
    """
    prod = 1
    for factor, power in factor_list:
        prod *= factor**power
    return prod


def _sumOfFactorsRecurse(prime_factors, result, depth, prod):
    if depth == len(prime_factors):
        result[0] += prod
        # print(f'add {prod}')
        return

    prime, power = prime_factors[depth]
    for i in range(power+1):
        _sumOfFactorsRecurse(prime_factors, result, depth+1, prod)
        prod *= prime


def sumOfPrimeFactorList(prime_factors):
    result = [0]
    _sumOfFactorsRecurse(prime_factors, result, 0, 1)
    return result[0]


def sum_factors(x):
    """
    Returns the sum of all factors. For example, the factors of
    12 are 1, 2, 3, 4, 6, and 12. Their sum is 28. sumFactors(12) = 28
    Throws an assertion error for non-positive integers.
    """
    return sumOfPrimeFactorList(prime_factorization(x))


sum_factors_cache: dict[int, int] = {}


def sum_factors2(n):
    # print(f'sumFactors({n}) = {sumFactors(n)} or ')

    if n in sum_factors_cache:
        # print(f'  re-use sum_factors_cache[{n}]')
        return sum_factors_cache[n]

    if n == 1:
        return 1
    
    # print(f'compute sumFactors({n})')

    def fixSF(k):
        if k < 0:
            return 0
        elif k == 0:
            return n
        else:
            return sum_factors2(k)
            # return sum_factors(k)

    s = 0
    i = 1
    while True:
        tmp = 3 * i*i
        k0 = n - (tmp - i)//2
        if k0 < 0:
            break
        k1 = n - (tmp + i)//2
        sign = (-1)**(i+1)
        
        sf0 = fixSF(k0)
        # print(f'[{n}] using sf({k0})={sf0}')
        sf1 = fixSF(k1)
        # print(f'[{n}] using sf({k1})={sf1}')

        # print(f'{sign:+d} * (s({k0}) + s({k1})) = {sign:+d} * ({sf0} + {sf1})')

        s += sign * (sf0 + sf1)
        
        i += 1

        # print(f' sf({n}) = {s}')

    if s != sum_factors(n):
        print(f'Error, sum_factors2({n}) is {sum_factors(n)}, but got {s}')
        sys.exit(1)
    sum_factors_cache[n] = s
    return s


def test_sum_factors(n):
    for i in range(2, n):
        # prime_factors = prime_factorization(i)
        # sum_factor_list = sumOfPrimeFactorList(prime_factors)
        sf1 = sum_factors(i)
        sf2 = sum_factors2(i)
        assert sf1 == sf2, \
            f'i={i}: {sf1} != {sf2}'
    print('ok')


def test_sum_factors2():
    end = 10000

    timer = time.perf_counter()
    s0 = sum([sum_factors(k) for k in range(1, end)])
    timer = time.perf_counter() - timer
    print(f'sum_factors {timer*1000:.3f} ms')

    timer = time.perf_counter()
    s1 = sum([sum_factors2(k) for k in range(1, end)])
    timer = time.perf_counter() - timer
    print(f'sum_factors2 {timer*1000:.3f} ms')

    print(s0, s1)
    
    # for n in range(1, 10000):
    #     a = sumFactors(n)
    #     b = sumFactors2(n)
    #     if a != b:
    #         print(f'ERROR sumFactors({n}) = {a} or {b}')
    #         break


def chinese_remainder_theorem(rules):
    """
    Returns t such that for all (m, a) in rules[], t % m == a
    """
    prod_all = list_prod([r[0] for r in rules])
    # print(f'{prod_all=}')

    soln = 0
    for m, a in rules:
        y = prod_all // m
        y_inv = pow(y, -1, m)
        soln += a * y * y_inv

    return soln % prod_all
    

def freq(seq):
    """
    Returns a list of (value, count) pairs of items from seq, ordered by
    decreasing count.
    For example, freq('foo') returns [('o': 2), ('f', 1)]
    """
    counter = collections.Counter(seq)
    frequency_list = list(counter.items())
    frequency_list.sort(key = lambda kv : -kv[1])
    return frequency_list


ints_re = re.compile(r'(-?\d+)')
                     
def ints(s):
    return [int(x) for x in ints_re.findall(s)]


def int_len(n):
    if n < 0:
        return 1 + int_len(-n)
    d = 1
    f = 10
    while n >= f:
        f *= 10
        d += 1
    return d


def count_set_bits_slow(x):
    if type(x) != int or x < 0:
        raise ValueError('Must be a non-negative integer')
    count = 0
    while x > 0:
        if x&1 == 1:
            count += 1
        x >>= 1
    return count


count_set_bits_byte_table: Optional[list[int]] = None

def count_set_bits(x):
    global count_set_bits_byte_table
    
    if count_set_bits_byte_table == None:
        count_set_bits_byte_table = [
            count_set_bits_slow(i) for i in range(256)]

    if type(x) != int or x < 0:
        raise ValueError('Must be a non-negative integer')

    count = 0
    while x > 0:
        count += count_set_bits_byte_table[x & 255]
        x >>= 8

    return count

        
def matrix_solve(matrix):
    """
    Take a matrix in the form:
    [x x x x
     x x x x 
     x x x x]
    and use row operations to convert it to:
    [1 0 0 z1
     0 1 0 z2
     0 0 1 z3]
  
    Each cell is an integer or a fractions.Fraction
    """
    height = len(matrix)
    width = len(matrix[0])
    assert height+1 == width
  
    def multRow(matrix, row_idx, factor):
        row = matrix[row_idx]
        for i in range(len(row)):
            row[i] *= factor
  
    def addRowMultiple(matrix, dest_row, src_row, factor):
        dest_row = matrix[dest_row]
        src_row = matrix[src_row]
        for i in range(len(dest_row)):
            dest_row[i] += factor * src_row[i]
  
    # print('before modification')
    # matrix_print(matrix)
  
    # for diag_idx in range(height):
    for diag_idx in range(height):
        # make the leading element 1 in this row
        factor = Fraction(1, matrix[diag_idx][diag_idx])
        multRow(matrix, diag_idx, factor)
        # print(f'simplify row {diag_idx}')
        # matrix_print(matrix)
    
        for row_id in range(height):
            if row_id == diag_idx:
                continue
    
            leading_value = matrix[row_id][diag_idx]
            factor = -leading_value
            addRowMultiple(matrix, row_id, diag_idx, factor)
            # print(f'row {row_id}')
            # matrix_print(matrix)


def matrix_print(matrix):
  width = len(matrix[0])
  def colMaxWidth(matrix, col):
    return max([len(str(row[col])) for row in matrix])
  col_widths = [colMaxWidth(matrix, c) for c in range(width)]
  for row in matrix:
    row_str = ' '.join([str(row[i]).rjust(col_widths[i]) for i in range(width)])
    print(row_str)


class DijkstraNode:
    """
    Skeleton implementation of a node for use with the Dijkstra class.
    """
    def __init__(self):
        self.cost = math.inf
        self.prev = None

    def __lt__(self, that):
        return self.cost < that.cost

    def adjacent(self, container):
        return []

    def is_end(self, container):
        False


def dijkstra(container, origin_node):
    """
    Runs Dijkstra's algorithm. Returns a list of nodes on an optimal
    route from the origin to an end node.

    node objects must support the following:
     - allow setting these attributes:
         cost - cost to reach this node from the origin
         prev - the node before this in an optimal route
         _eheap_index - used by heap internally
     - a __lt__ implementation that compares costs
     - adjacent(container) method, returning an iterable of (node, cost)
       tuples, where the cost is the incremental cost to travel from self
       to this node.
    
     - is_end(container) returning True iff this is an end node

    node.prev == None iff node has not been added to the heap
    
    The container object is passed to node.adjacent() and node.is_end().
    """
    
    heap = eheap.Heap()
    origin_node.cost = 0
    heap.add(origin_node)
    end_node = None
    count = 0

    # print('start dijkstra')
        
    while True:
        node = heap.pop()
        if not node:
            # print('heap empty')
            break

        # print(f'pop {node}')

        count += 1
        # container.dense[node.row][node.col] = 'b'
        # draw_grid.drawGrid(container.dense, f'day13.{count:03}.png')
        
        # if count == 100:
        #     break

        # print(f'pop {node}')
        
        if node.is_end(container):
            # print('end found')
            end_node = node
            break

        # print(f'from {node}')
        for (adj_node, cost) in node.adjacent(container):
            # print('  ' + str(adj_node))
            prev_cost = adj_node.cost
            new_cost = node.cost + cost
            if new_cost < adj_node.cost:
                adj_node.cost = new_cost
                adj_node.prev = node
                if math.isinf(prev_cost):
                    heap.add(adj_node)
                    # print(f'  new node {adj_node}')
                else:
                    heap.decrease_key(adj_node)
                    # print(f'  shortcut to {adj_node}')

    if not end_node:
        raise Exception('Dijkstra\'s algorithm failed to find an end node')

    route = [end_node]
    node = end_node
    while node.prev:
        node = node.prev
        route.append(node)

    route.reverse()

    return route
        

def md5(s, suffix = None):
    if suffix != None:
        s = s + str(suffix)
    return hashlib.md5(bytes(s, 'ascii')).hexdigest()
      
  
if __name__ == '__main__':
    # testPasteGrid()
    # testSparseRead()
    # print(repr(freq('google')))
    # print(repr(freq(['cow', 'dog', 'cat', 'horse', 'dog', 'horse', 'dog', 'dog'])))
    # testClump()
    test_sum_factors2()
    pass
