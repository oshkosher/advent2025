#!/usr/bin/env python3


"""
Advent of Code 2025, Day 10: Factory

Integer programming with boolean matrices

Use UV to run with threaded interpreter:
  uv run -p 3.14t python day10.py

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

"""
uv run -p 3.14t python ./day10.py
uv run --with viztracer -p 3.14t python -m viztracer --log_sparse -o parallel_machines_without_gil.json day10.py

total time spent solving problems. nf = number of free variables
time is in microseconds

nf  count     usec
 0     71    18063
 1     46     6830
 2     29    12146
 3      8   310078

The 8 machines with 3 free variables take 89% of the time.  They are
also the problems with the most buttons (the 5 hardest are the only
problems with 13 or more buttons).

There's not much to be gained by each machine to one thread, because
there will be a huge load imbalance where a few unlucky threads get
the few huge machines. It would be better to have one thread do all the
easy machines, and spread each huge machine across multiple threads.

To get the compute threads working on huge machines as quickly as possible,
we could order the list by decreasing button count. That can be determined
instantly, unlike computing the number of free variables. 

The main thread could compute small (n_free_variables < 3) machines
directly, and send any large machines to a compute queue.
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools
from math import floor, ceil

# from viztracer import log_sparse, get_tracer

# linear programming library
# import pulp

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


class Machine:
    def __init__(self, idx, lights, buttons, button_bits, jolts):
        self.idx = idx
        self.lights = lights

        # list of tuples, just like the input
        self.buttons = buttons

        # list of integers with the given bits set
        # i.e. (0,1,2,3) -> 15
        self.button_bits = button_bits
        
        self.jolts = jolts

    def __repr__(self):
        return f'Machine({self.idx}, {len(self.jolts)}x{len(self.buttons)})'

    line_re = re.compile(r'\[([#.]*)\] (.*) {(.*)}')
        
    @staticmethod
    def parse(line, idx = None):
        """
        Parse line like this:
          [.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
        into a Machine object.
        The first part [.##.] is a bit mask: 0110
        The second part '(3) (1,3), ...' is a bit mask for the part 1
          problem, and a list of indices for part 2.
        The third part {3,5,4,7} is a list of joltage values for part 2.
        """
        match = Machine.line_re.match(line)
        assert match
        light_str, buttons_str, jolts_str = match.groups()

        bit = 1
        lights = 0
        for c in light_str:
            if c == '#':
                lights += bit
            bit *= 2

        def button_positions_to_int(positions):
            return sum([2**x for x in positions])
            
        button_ints = [tuple([int(v) for v in b[1:-1].split(',')])
                       for b in buttons_str.split(' ')]
        button_bits = [button_positions_to_int(b) for b in button_ints]

        jolts = [int(x) for x in jolts_str.split(',')]

        return Machine(idx, lights, button_ints, button_bits, jolts)
    

class VizEventNameOnly:
    def __init__(
            self, tracer: "VizTracer", event_name: str
    ) -> None:
        self._tracer = tracer
        self._name = event_name
        self._start = 0

    def __enter__(self) -> None:
        if not self._tracer: return
        self._start = self._tracer.getts()

    def __exit__(self, type, value, trace) -> None:
        if not self._tracer: return
        
        dur = self._tracer.getts() - self._start
        raw_data = {
            "ph": "X",
            "ts": self._start,
            "name": self._name,
            "dur": dur,
            "cat": "FEE",
        }
        self._tracer.add_raw(raw_data)


def log_event_name_only(event_name: str) -> VizEventNameOnly:
    return VizEventNameOnly(get_tracer(), event_name)
    

def light_button_count_bfs(machine):
    # each tuple on the queue:
    # (current_lights, list of indices of buttons pressed)
    q = collections.deque()
    q.append((0, []))

    while len(q) > 0:
        current_lights, buttons_pressed = q.popleft()
        if current_lights == machine.lights:
            return buttons_pressed

        next_bi = 0 if len(buttons_pressed)==0 else buttons_pressed[-1]
        for bi in range(next_bi, len(machine.button_bits)):
            next_lights = current_lights ^ machine.button_bits[bi]
            pressed = buttons_pressed + [bi]
            q.append((next_lights, pressed))

    return None


def part1(machines):
    button_press_total = 0
    for i, m in enumerate(machines):
        pressed_list = light_button_count_bfs(m)
        button_press_total += len(pressed_list)

    print(button_press_total)


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


def array_swap(a, i1, i2):
    tmp = a[i1]
    a[i1] = a[i2]
    a[i2] = tmp


def largest_axis_last(free_maxes, vectors):
    """
    Rearrange the columns so the column with the largest free_max value
    goes last, because with the trim_search_range() logic, the last column
    is the fastest
    """
    n = len(free_maxes)
    if n < 2:
        return
    
    lc = 0  # largest column
    for i in range(1, n):
        if free_maxes[i] > free_maxes[lc]:
            lc = i
    if lc == n-1:
        return
    
    array_swap(free_maxes, lc, n-1)
    array_swap(vectors, lc+1, n)


def compute_solution_vectors(A, scale, free_cols):
    """
    A: RREF augmented matrix
    free_cols: list of free columns

    Returns a list of len(free_cols)+1 vectors
      vector 0: base vector
      vectors 1..len(free_cols): free vectors

    All solution vectors are linear combinations of base and the free vectors.
    For example: result[0] + 2*result[1] + 5*result[2]
    """

    # length of each vector == length of solution vector
    n = A.width - 1

    n_empty_rows = A.count_tail_zero_rows()
    n_non_empty_rows = A.height - n_empty_rows

    vectors = [[0] * n for _ in range(len(free_cols)+1)]

    # compute base vector
    last_col = A.column(-1)
    base = vectors[0]
    wi = 0  # write index
    for ri in range(n_non_empty_rows):
        while wi in free_cols:
            base[wi] = 0
            wi += 1
        base[wi] = last_col[ri]
        wi += 1

    for vi, c in enumerate(free_cols):
        assert A.height <= A.width-1

        v = vectors[vi+1]

        read_r = 0
        for r in range(n):
            if r in free_cols:
                if r == c:
                    v[r] = scale
                else:
                    v[r] = 0
            else:
                v[r] = -A[read_r][c]
                read_r += 1

        # print(f'free_vec[{c}] = {v!r}')

    return vectors


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


def vec_add_mult_sum_divisible(v, dv, k, divisor):
    """
    Checks if all entries in (v + k dv) are divisible by divisor.
    If so, return their sum / divisor.
    Otherwise return None.
    """
    if divisor == 1:
        # return vec_add_mult_sum_ints(v, dv, k)
        s = 0
        for i in range(len(v)):
            s += v[i] + k * dv[i]
        return s
    
    s = 0
    for i in range(len(v)):
        x = v[i] + k * dv[i]
        # print(f'  v[{i}] = {x}')
        if x % divisor != 0:
            return None
        s += x

    return s // divisor
        

def solve_free1(max_value, base_vec, free_vec, scale):
    """
    Solve a system with one free variable.
    base_vec: null space base vector
    free_vec: null space free vector. The null space is all vectors of the
      form base_vec + k * free_vec

    Depending on whether the sum of free_vec is positive or negative,
    count up from 0 or down from max_value. Return the first solution that
    is all non-negative and integers.
    """
    
    search_range = trim_search_range(base_vec, free_vec, 0, max_value)
    for k in search_range:
        s = vec_add_mult_sum_divisible(base_vec, free_vec, k, scale)
        if s:
            return s
        
    return math.inf


def solve_free2(max_values, vectors, scale):
    """
    Solve a system with two free variables.
    """
    n = len(vectors[0])
    w = [0] * n
    
    best_sum = math.inf

    for r in range(max_values[0]+1):
        add_scaled_vector(w, vectors[0], vectors[1], r)
        search_range = trim_search_range(w, vectors[2], 0, max_values[1])
        
        for c in search_range:
            s = vec_add_mult_sum_divisible(w, vectors[2], c, scale)
            if s is not None:
                if s < best_sum:
                    best_sum = s
                break
                
    return best_sum


def solve_free3(max_values, vectors, scale):
    """
    Solve a system with three free variables.
    The input doesn't contain anything larger than three.
    """
    assert len(vectors) == 4 and len(max_values) == 3
    v0 = [0] * len(vectors[0])
    v1 = [0] * len(vectors[0])

    best_sum = math.inf

    for k0 in range(max_values[0]+1):
        add_scaled_vector(v0, vectors[0], vectors[1], k0)
        for k1 in range(max_values[1]+1):
            add_scaled_vector(v1, v0, vectors[2], k1)
            
            search_range = trim_search_range(v1, vectors[3], 0, max_values[2])

            for k2 in search_range:
                s = vec_add_mult_sum_divisible(v1, vectors[3], k2, scale)
                if s:
                    if s < best_sum:
                        best_sum = s
                    break
        
    return best_sum


def gen_matrix_denominators(matrix):
    """
    Generates all the denominators of elements in matrix.
    """
    for row in matrix:
        for cell in row:
            if isinstance(cell, MyFraction):
                yield cell.denominator
            else:
                yield 1

                
def make_matrix_integers(matrix):
    """
    If the matrix contains any MyFraction values, scale up the matrix
    by the least common multiple of their denominators and return that LCM.
    Otherwise do nothing and return 1.
    """
    lcm = math.lcm(*list(gen_matrix_denominators(matrix)))
    if lcm != 1:
        # print(f'scale by {lcm}')
        for row in matrix:
            for i in range(len(row)):
                row[i] = int(row[i] * lcm)
    return lcm


def joltage_presses_init(machine):
    """
    Start the computation, to the point where the number of free variables
    is computed.

    If the elments in A was multiplied by a scale factor to make every
    entry an integer, scale will be that value, otherwise 1.

    returns A, scale, free_cols, free_maxes
    """
    # create a matrix representing what lights up when each
    # button is pressed
    A = Matrix(height = len(machine.jolts),
               width = len(machine.buttons) + 1)
    for c, button in enumerate(machine.buttons):
        for r in button:
            A[r][c] = 1

    n_vars = len(machine.buttons)
    soln_max_values = max_values(A, machine.jolts)

    # augment the matrix with the goal values
    for r, jolts in enumerate(machine.jolts):
        A[r][n_vars] = jolts
        
    A.row_reduce()

    # remove empty rows at the bottom
    n_zero_rows = A.count_tail_zero_rows()
    A.remove_rows(A.height - n_zero_rows, n_zero_rows)

    # scale = 1
    scale = make_matrix_integers(A)

    # list free columns
    free_cols = find_free_var_columns(A)

    # subset of soln_max_values just for the free columns
    # this is our search space
    free_maxes = [soln_max_values[free_cols[i]]
                  for i in range(len(free_cols))]
        
    return A, scale, free_cols, free_maxes


def joltage_presses_finish(machine, A, scale, free_cols, free_maxes):
    """
    Compute the null space and solve the problem. 
    """

    # compute the null space
    vectors = compute_solution_vectors(A, scale, free_cols)

    largest_axis_last(free_maxes, vectors)
    
    n_free = len(free_cols)
    n_vars = len(machine.buttons)

    if n_free == 0:
        soln = A.column(-1)[:n_vars]
        soln_size = sum(soln)
    elif n_free == 1:
        soln_size = solve_free1(free_maxes[0], vectors[0], vectors[1], scale)
    elif n_free == 2:
        soln_size = solve_free2(free_maxes, vectors, scale)
    else:
        soln_size = solve_free3(free_maxes, vectors, scale)

    return soln_size
    
        
def part2(machines, verbose = False):

    if verbose:
        print('\t'.join(['idx', 'n_free', 'mx_size', 'n_press']))
    
    total_presses = 0
    for i, machine in enumerate(machines):

        # with log_event_name_only(f'#{i}'):
        
        A, scale, free_cols, free_maxes = joltage_presses_init(machine)
        press_count = joltage_presses_finish(machine, A, scale,
                                             free_cols, free_maxes)
        total_presses += press_count

        if verbose:
            n_free = len(free_cols)
            matrix_size = f'{len(machine.jolts)}x{len(machine.buttons)}'
            print(f'{i}\t{n_free}\t{matrix_size}\t{press_count}')
        
    print(total_presses)
    

def part2_time_each_problem(machines, iter_count = 10):
    """
    Solve each problem (iter_count) times, and use the best time.
    Output the problems in increasing solve time with these columns:
    solve_time, cumulative_time, n_free_variables, machine_idx, matrix_size
    """
    total_presses = 0
    best_times_sum = 0

    print('\t'.join(['idx', 'usec', 'n_free', 'mx_size', 'n_press']))
        
    for i, machine in enumerate(machines):

        best_time = math.inf
        for _ in range(iter_count):
            nanos = time.perf_counter_ns()

            A, scale, free_cols, free_maxes = joltage_presses_init(machine)
            press_count = joltage_presses_finish(machine, A, scale, free_cols,
                                                 free_maxes)
            n_free = len(free_cols)
            
            nanos = time.perf_counter_ns() - nanos
            if nanos < best_time:
                best_time = nanos

        total_presses += press_count
        matrix_size = f'{len(machine.jolts)}x{len(machine.buttons)}'
        best_times_sum += best_time
        
        print(f'{i}\t{best_time//1000}\t{n_free}'
              f'\t{matrix_size}\t{press_count}')
        sys.stdout.flush()

    print(f'sum of best times: {best_times_sum/1e6:.2f} ms')

    print(total_presses)

    
def main(args):
    filename = 'day10.in'
    n_threads = 3

    if len(args) > 1:
        print('Extra command line arguments (expecting a filename or nothing)')
        return 1
    
    if len(args) > 0:
        filename = args[0]

    # test_fractions()
            
    # read input as a list of strings
    input = read_problem_input(filename)
    machines = [Machine.parse(line, idx) for idx, line in enumerate(input)]
  
    t0 = time.perf_counter_ns()
    part1(machines)
    t1 = time.perf_counter_ns()
    # part2_time_each_problem(machines, 20)
    part2(machines, False)
    t2 = time.perf_counter_ns()
    print(f'part1 {(t1-t0)/1e6:.2f} millis')
    print(f'part2 {(t2-t1)/1e6:.2f} millis')
    
    

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
