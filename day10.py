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
# import numpy as np
from math import floor, ceil

import threading
from queue import Queue

from viztracer import log_sparse, get_tracer

# linear programming library
# import pulp

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *

# import day10_solver


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

        # lock, search_size, search_done, and best_soln will be set if
        # multiple thread are used to solve the machine
        self.lock = None

        # will be set in parallel solver
        self.base_vec = None
        self.free_vecs = None
        
        # The problem is parallelized by splitting the search space into
        # multiple ranges. For example, if the first free variable is
        # in the range [0..30], the chunks [0..10], [11..30] could be
        # assigned to two different threads. The total size (not the number
        # of chunks) will be stored in search_size. In this case, 31.
        #
        # This value is set before the problem is assigned to threads,
        # and does not change.
        self.search_size = None

        self.search_chunk_size = None

        # this tracks how much of the search space has been processed.
        # For example, if the search space is [0..30] and the chunks
        # are [0..10], [11..30], when [0..10] is completed search_done
        # will be incrased by 11, and when [11..30] is completed
        # search_done will be incrased by 20.
        #
        # This value is protected by self.lock
        self.search_done = None

        # After a thread processes a chunk, if it found a solution better
        # than best_soln, it will update best_soln.
        #
        # This value is protected by self.lock
        self.best_soln = None

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


    def init_threaded_soln(self, free_maxes, n_threads):
        self.lock = threading.Lock()
        self.search_size = free_maxes[0] + 1

        """
        How big should the work chunks be? Larger is more efficient, smaller
        provides for better load balancing. Let's start with a cap
        of n_threads * 3, so on average each thread will do 3 pieces of
        work.
        """
        thread_factor = 3
        chunk_count = n_threads * thread_factor

        if self.search_size <= chunk_count:
            self.search_chunk_size = 1
        else:
            # round up, so when chunk_count doesn't evenly divide search_size,
            # most of the chunks are large and the last one is small.
            # For example, with search_size = 10 and chunk_count=4,
            # this will yield a chunk size of (10+3)//4 = 3 and chunks
            # of size [3, 3, 3, 1].
            # If we rounded down the chunk size would be 10//4 = 2 and we'd get
            # chunks of size [2, 2, 2, 2, 2], exceeding chunk_count.
            self.search_chunk_size = ((self.search_size + chunk_count - 1)
                                      // chunk_count)
        
        self.search_done = 0
        self.best_soln = math.inf

    def search_chunk_done(self, chunk_size, best_soln):
        """
        Returns true if the search is complete.
        """
        with self.lock:
            self.search_done += chunk_size
            # print(f'machine {self.idx} {self.search_done}/{self.search_size} done')
            if best_soln < self.best_soln:
                self.best_soln = best_soln
            is_done = self.search_done == self.search_size
        return is_done
    
    

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


def joltage_presses_using_pulp(machine):
    """
    Use the Pulp linear programming library to solve the problem.
    This was my original solution, but I wrote a new solver from scratch
    that is about 20x faster.

    Find a vector x such that A x = b in an underconstrained system,
    looking for a minimum sum of the elements in vector x.
    Also, the input matrix is all 0's and 1's.
    
    [0 0 0 0 1 1  * [1 = [3
     0 1 0 0 0 1     3    5
     0 0 1 1 1 0     0    4
     1 1 0 1 0 0]    3    7]
                     1    
                     2]
    """

    height = len(machine.jolts)
    width = len(machine.buttons)

    A_ints = [[0] * width for _ in range(height)]

    A = np.zeros((height, width))
    for c in range(width):
        button = machine.buttons[c]
        for r in button:
            A[r][c] = 1

            A_ints[r][c] = 1

    goal = np.array(machine.jolts)

    prob = pulp.LpProblem("find_x", pulp.LpStatusOptimal)
    x_vars = [pulp.LpVariable(f"x{i}", lowBound=0, cat="Integer")
              for i in range(width)]

    # equality constraints
    for i in range(height):
        prob += pulp.lpSum(A[i,j]*x_vars[j] for j in range(width)) == goal[i]

    # find smallest solution
    prob += pulp.lpSum(x_vars)

    timer = time.perf_counter_ns()
    # prob.solve()    # default CBC solver
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    timer = time.perf_counter_ns() - timer

    if pulp.LpStatus[prob.status] != "Optimal":
        print("no integer solution")
        return None
    
    soln_float = [v.value() for v in x_vars]
    soln = [round(v) for v in soln_float]

    # print(f'problem_list.append( (\n{A_ints!r},\n  {machine.jolts!r},\n  {soln!r}\n))')
    
    # soln_vector = np.array(soln)
    # print(f'A x = {A @ soln_vector}')
    # check_float = [v for v in A @ soln_vector]
    # check = [round(v) for v in check_float]
    # if check != machine.jolts:
    #     print(f'ERROR, goal={machine.jolts}, result={check} or {check_float}')
    # print(repr(check))

    return sum(soln)


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
        

def solve_free1(max_value, base_vec, free_vec):
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
        s = vec_add_mult_sum_ints(base_vec, free_vec, k)
        if s:
            return s
        
    return 0


def solve_free2_visual(max_values, base_vec, free_vecs):
    """
    Given a problem with two free variables, draw a 2-d grid of
    solutions and non-solutions.
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


def solve_free2(max_values, base_vec, free_vecs):
    """
    Solve a system with two free variables.
    """
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


def solve_free3(max_values, base_vec, free_vecs):
    """
    Solve a system with three free variables.
    The input doesn't contain anything larger than three.
    """
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


def joltage_presses(machine, n_free_vars_arg = None):

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

    # list free columns
    free_cols = find_free_var_columns(A)
    n_free = len(free_cols)
    if n_free_vars_arg is not None:
        n_free_vars_arg[0] = n_free

    # compute the null space
    # every vector of the form base_vec + k * free_vecs[i]
    # is a solution to the matrix
    base_vec, free_vecs = compute_solution_vectors(A, free_cols)
        
    assert len(free_vecs) == n_free

    # subset of soln_max_values just for the free columns
    # this is our search space
    free_maxes = [soln_max_values[free_cols[i]] for i in range(n_free)]

    # specialize the solution for how many free variables there are
    if n_free == 0:
        soln = A.column(-1)[:n_vars]
        soln_size = sum(soln)
    elif n_free == 1:
        soln_size = solve_free1(free_maxes[0], base_vec, free_vecs[0])
    elif n_free == 2:
        soln_size = solve_free2(free_maxes, base_vec, free_vecs)
    else:
        soln_size = solve_free3(free_maxes, base_vec, free_vecs)

    return soln_size


def joltage_presses_init(machine):
    """
    Start the computation, to the point where the number of free variables
    is computed.

    returns A, free_cols, free_maxes
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

    # list free columns
    free_cols = find_free_var_columns(A)

    # subset of soln_max_values just for the free columns
    # this is our search space
    free_maxes = [soln_max_values[free_cols[i]]
                  for i in range(len(free_cols))]
        
    return A, free_cols, free_maxes


def joltage_presses_finish(machine, A, free_cols, free_maxes):
    """
    Compute the null space and solve the problem. 
    """

    # compute the null space
    # every vector of the form base_vec + k * free_vecs[i]
    # is a solution to the matrix
    base_vec, free_vecs = compute_solution_vectors(A, free_cols)

    n_free = len(free_cols)
    n_vars = len(machine.buttons)

    if n_free == 0:
        soln = A.column(-1)[:n_vars]
        soln_size = sum(soln)
    elif n_free == 1:
        soln_size = solve_free1(free_maxes[0], base_vec, free_vecs[0])
    elif n_free == 2:
        soln_size = solve_free2(free_maxes, base_vec, free_vecs)
    else:
        soln_size = solve_free3(free_maxes, base_vec, free_vecs)

    return soln_size

        
def part2(machines):
    total_presses = 0
    for i, machine in enumerate(machines):
        with log_event_name_only(f'#{i}'):
            total_presses += joltage_presses(machine)
    print(total_presses)


class ComputeThread(threading.Thread):
    def __init__(self, thread_id, todo_q, results_q):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.todo_q = todo_q
        self.results_q = results_q

        # [(idx, usec, n_free, mx_size, n_press), ...]
        self.results_table = []

    def run(self):
        # self.name = f'Thread {self.thread_id}'
        getting_ns = 0
        putting_ns = 0
        jolting_ns = 0
        tasks_done = 0
        n_free_vars = [0]

        while (True):
            timer = time.perf_counter_ns()
            machine = self.todo_q.get()
            if tasks_done > 0:
                getting_ns += time.perf_counter_ns() - timer
            
            # all done
            if machine == 0:
                # put it back on the queue so other workers will see it
                self.todo_q.put(0)
                break

            nanos = time.perf_counter_ns()

            with log_event_name_only(f'#{machine.idx}'):
                press_count = joltage_presses(machine, n_free_vars)
            
            nanos = time.perf_counter_ns() - nanos
            jolting_ns += nanos
            
            tasks_done += 1

            matrix_size = f'{len(machine.jolts)}x{len(machine.buttons)}'

            result_tup = (machine.idx, nanos // 1000, n_free_vars[0],
                          matrix_size, press_count)
            
            timer = time.perf_counter_ns()
            # self.results_q.put(press_count)
            self.results_q.put(result_tup)
            putting_ns += time.perf_counter_ns() - timer
            

        getting_ms = f'{getting_ns / 1e6:.3f}'
        putting_ms = f'{putting_ns / 1e6:.3f}'
        jolting_ms = f'{jolting_ns / 1e6:.0f}'
        print(f'thread {self.thread_id}: {tasks_done} tasks, {jolting_ms}ms computing, {getting_ms}ms getting, {putting_ms}ms putting')


class ComputeThreadMultipleReps(threading.Thread):
    """
    Like ComputeThread, but each machine is solved multiple times and
    the fastest time is saved as the solution time.
    """
    def __init__(self, thread_id, todo_q, results_q, rep_count):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.todo_q = todo_q
        self.results_q = results_q
        self.rep_count = rep_count

        # [(idx, usec, n_free, mx_size, n_press), ...]
        self.results_table = []

    def run(self):
        getting_ns = 0
        putting_ns = 0
        jolting_ns = 0
        tasks_done = 0
        n_free_vars = [0]

        while (True):
            timer = time.perf_counter_ns()
            machine = self.todo_q.get()
            if tasks_done > 0:
                getting_ns += time.perf_counter_ns() - timer
            
            # all done
            if machine == 0:
                # put it back on the queue so other workers will see it
                self.todo_q.put(0)
                break

            soln_ns = math.inf
            for _ in range(self.rep_count):
                timer = time.perf_counter_ns()
                press_count = joltage_presses(machine, n_free_vars)
                timer = time.perf_counter_ns() - timer
                if timer < soln_ns:
                    soln_ns = timer

            jolting_ns += soln_ns
            tasks_done += 1

            matrix_size = f'{len(machine.jolts)}x{len(machine.buttons)}'

            results = (machine.idx, soln_ns // 1000, n_free_vars[0],
                          matrix_size, press_count)
            
            timer = time.perf_counter_ns()
            # self.results_q.put(press_count)
            self.results_q.put(results)
            putting_ns += time.perf_counter_ns() - timer
            

        getting_ms = f'{getting_ns / 1e6:.3f}'
        putting_ms = f'{putting_ns / 1e6:.3f}'
        jolting_ms = f'{jolting_ns / 1e6:.0f}'
        print(f'thread {self.thread_id}: {tasks_done} tasks, {jolting_ms}ms computing, {getting_ms}ms getting, {putting_ms}ms putting')
    

def part2_threaded(machines, n_compute_threads = 4):
    t0 = time.perf_counter_ns()
    todo_q = Queue()
    results_q = Queue()
    result_table = []
    
    # 1300-1800 usec
    with log_event_name_only('start threads'):
        t1 = time.perf_counter_ns()
        compute_threads = []
        for i in range(n_compute_threads):
            t = ComputeThread(i, todo_q, results_q)
            t.name = f'Thread {i}'
            # t = ComputeThreadMultipleReps(i, todo_q, results_q, 10)
            t.start()
            compute_threads.append(t)
        t2 = time.perf_counter_ns()

    # 300-400 usec
    with log_event_name_only('enqueue jobs'):
        nanos = time.perf_counter_ns()
        for machine in machines:
            todo_q.put(machine)
        todo_q.put(0)
        t3 = time.perf_counter_ns()

    # 750000 - 850000 usec
    with log_event_name_only('wait for results'):
        total_presses = 0
        for i in range(len(machines)):
            result = results_q.get()
            total_presses += result[-1]
            result_table.append(result)
        nanos = time.perf_counter_ns() - nanos
        t4 = time.perf_counter_ns()

    print(f'{total_presses} in {nanos / 1e6:.3f} ms')

    # 1400-5000 usec
    for t in compute_threads:
        t.join()

    # print('flock gathered')
    t5 = time.perf_counter_ns()

    print(f'{(t1-t0) / 1e3} us: create queues')
    print(f'{(t2-t1) / 1e3} us: create threads')
    print(f'{(t3-t2) / 1e3} us: enqueue machines')
    print(f'{(t4-t3) / 1e3} us: gather results')
    print(f'{(t5-t4) / 1e3} us: join')

    """
    result_table.sort()
    print('\t'.join(['idx', 'usec', 'n_free', 'mx_size', 'n_press']))
    for idx, usec, n_free, mx_size, n_press in result_table:
        print(f'{idx}\t{usec}\t{n_free}\t{mx_size}\t{n_press}')
    """
    total_press = sum([row[4] for row in result_table])
    print(total_press)
    

def part2_time_each_problem(machines, iter_count = 10):
    """
    Solve each problem (iter_count) times, and use the best time.
    Output the problems in increasing solve time with these columns:
    solve_time, cumulative_time, n_free_variables, machine_idx, matrix_size
    """
    total_presses = 0
    n_free_holder = [0]
    table = []

    print('\t'.join(['idx', 'usec', 'n_free', 'mx_size', 'n_press']))
        
    for i, machine in enumerate(machines):
        # nanos = time.perf_counter_ns()
        # press_count = joltage_presses(machine, n_free_vars)

        # press_count = joltage_presses_split(machine)
        # A, free_cols, free_maxes = joltage_presses_init(machine)
        # press_count = joltage_presses_finish(machine, A, free_cols, free_maxes)
        # n_free_vars[0] = len(free_cols)
        
        # total_presses += press_count
        # best_time = time.perf_counter_ns() - nanos

        best_time = math.inf
        for _ in range(iter_count):
            nanos = time.perf_counter_ns()
            
            # press_count = joltage_presses(machine, n_free_holder)
            # n_free = n_free_holder[0]

            A, free_cols, free_maxes = joltage_presses_init(machine)
            press_count = joltage_presses_finish(machine, A, free_cols,
                                                 free_maxes)
            n_free = len(free_cols)
            
            nanos = time.perf_counter_ns() - nanos
            if nanos < best_time:
                best_time = nanos

        total_presses += press_count
        matrix_size = f'{len(machine.jolts)}x{len(machine.buttons)}'
        
        print(f'{i}\t{best_time//1000}\t{n_free}'
              f'\t{matrix_size}\t{press_count}')
        sys.stdout.flush()
        
        # table.append([best_time, n_free, i, matrix_size])
        
        # sys.stdout.write(f'\rmachine {i} of {len(machines)}')
        
    # sys.stdout.write('\n')
    # table.sort()

    """
    cumulative_us = 0
    print('time_us\tcum_us\tn_free\tidx\tmx_size')
    for row in table:
        row[0] = row[0] // 1000
        cumulative_us += row[0]
        row[1:1] = [cumulative_us]
        print('\t'.join([str(x) for x in row]))
       """
    
    print(total_presses)


class Timer:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total_ns = 0
        self.shortest_ns = math.inf
        self.start_ns = None

    def start(self):
        self.start_ns = time.perf_counter_ns()

    def end(self):
        assert self.start_ns is not None
        nanos = time.perf_counter_ns() - self.start_ns
        self.start_ns = None
        self.count += 1
        self.total_ns += nanos
        if nanos < self.shortest_ns:
            self.shortest_ns = nanos

    def report(self):
        print(f'{self.name}: {self.shortest_ns / 1000:.0f} usec')


class SolutionTimers:
    def __init__(self):
        self.create_matrix = Timer('create matrix')
        self.row_reduce = Timer('row reduce')
        self.trim = Timer('trim')
        self.null_space = Timer('null space')
        self.search = Timer('search')

    def report(self):
        self.create_matrix.report()
        self.row_reduce.report()
        self.trim.report()
        self.null_space.report()
        self.search.report()

        
def joltage_presses_instrumented(machine, timers):
    """
    Testing from machine 23, a 10x13 with 3 free variables
    create matrix: 31 usec
    row reduce: 939 usec
    trim: 2 usec
    null space: 28 usec
    search: 313184 usec
    """

    timers.create_matrix.start()
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

    timers.create_matrix.end()
        
    timers.row_reduce.start()
    A.row_reduce()
    timers.row_reduce.end()

    timers.trim.start()
    # remove empty rows at the bottom
    n_zero_rows = A.count_tail_zero_rows()
    A.remove_rows(A.height - n_zero_rows, n_zero_rows)
    timers.trim.end()

    timers.null_space.start()
    # list free columns
    free_cols = find_free_var_columns(A)
    n_free = len(free_cols)

    # compute the null space
    # every vector of the form base_vec + k * free_vecs[i]
    # is a solution to the matrix
    base_vec, free_vecs = compute_solution_vectors(A, free_cols)
    timers.null_space.end()
        
    assert len(free_vecs) == n_free

    # subset of soln_max_values just for the free columns
    # this is our search space
    free_maxes = [soln_max_values[free_cols[i]] for i in range(n_free)]

    timers.search.start()
    # specialize the solution for how many free variables there are
    if n_free == 0:
        soln = A.column(-1)[:n_vars]
        soln_size = sum(soln)
    elif n_free == 1:
        soln_size = solve_free1(free_maxes[0], base_vec, free_vecs[0])
    elif n_free == 2:
        soln_size = solve_free2(free_maxes, base_vec, free_vecs)
    else:
        soln_size = solve_free3(free_maxes, base_vec, free_vecs)
    timers.search.end()

    return soln_size



def part2_single_problem(machine, n_reps):
    timers = SolutionTimers()

    for _ in range(n_reps):
        joltage_presses_instrumented(machine, timers)

    timers.report()


def solve_free3_partial(max_values, base_vec, free_vecs,
                        axis0_range):
    """
    Solve a system with three free variables.
    The input doesn't contain anything larger than three.
    Only check the axis0_range values for the first free variable.
    """
    assert len(free_vecs) == 3 and len(max_values) == 3
    v0 = [0] * len(base_vec)
    v1 = [0] * len(base_vec)

    best_sum = math.inf

    # for k0 in range(max_values[0]+1):

    for k0 in axis0_range:
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


class ComputeThreadPartialMachines(threading.Thread):
    def __init__(self, thread_id, todo_q, results_q):
        threading.Thread.__init__(self)
        self.name = f'Thread {thread_id}'
        self.thread_id = thread_id
        self.todo_q = todo_q
        self.results_q = results_q

        # [(idx, usec, n_free, mx_size, n_press), ...]
        self.results_table = []

    def run(self):
        # self.name = f'Thread {self.thread_id}'
    
        while (True):
            task = self.todo_q.get()
            
            # all done
            if task == 0:
                # put it back on the queue so other workers will see it
                self.todo_q.put(0)
                break

            machine, matrix, free_cols, free_maxes, rng = task
            
            with log_event_name_only(f'#{machine.idx}@{rng.start}-{rng.stop}'):
                press_count = solve_free3_partial(free_maxes, machine.base_vec,
                                                  machine.free_vecs, rng)
            
            
            # print(f'[{self.thread_id}] machine {machine.idx} {rng} '
            #       f'best={press_count}')

            # matrix_size = f'{len(machine.jolts)}x{len(machine.buttons)}'

            if machine.search_chunk_done(len(rng), press_count):
                # print(f'[{self.thread_id}] machine {machine.idx} done')
                self.results_q.put(machine)


def enqueue_large_machine(large_machine_q, machine, n_threads, matrix,
                          free_cols, free_maxes):
    """
    Initialize a Machine object to be computed in parallel, and create
    multiple tasks on the job queue to compute it.
    """
    machine.init_threaded_soln(free_maxes, n_threads)

    machine.base_vec, machine.free_vecs = \
        compute_solution_vectors(matrix, free_cols)

    end = free_maxes[0]
    for range_start in range(0, end+1, machine.search_chunk_size):
        range_end = min(end+1, range_start + machine.search_chunk_size)
        rng = range(range_start, range_end)
        task = (machine, matrix, free_cols, free_maxes, rng)
        large_machine_q.put(task)
        # print(f'enqueue machine {machine.idx} space={free_maxes!r} '
        #       f'range={rng!r}')
    

def part2_threaded_large_ones(machines, n_threads):
    total_sum = 0
    large_machine_q = Queue()
    completed_machine_q = Queue()

    print(f'using {n_threads} compute threads')

    compute_threads = []
    for i in range(n_threads):
        t = ComputeThreadPartialMachines(i, large_machine_q,
                                         completed_machine_q)
        t.start()
        compute_threads.append(t)

    # see if this speeds things up; it moves the largest problems to the
    # front of the list so they get quickly farmed out to the compute threads
    # machines.sort(key = lambda m: -len(m.buttons))

    n_offloaded = 0
    for machine in machines:
        matrix, free_cols, free_maxes = joltage_presses_init(machine)

        # put anything with more than two free columns on the large_machine_q
        if len(free_cols) > 2:
            enqueue_large_machine(large_machine_q, machine, n_threads, matrix,
                                  free_cols, free_maxes)
            n_offloaded += 1
        else:
            # compute directly
            with log_event_name_only(f'#{machine.idx}'):
                press_count = joltage_presses_finish(machine, matrix, free_cols,
                                                     free_maxes)
            # print(f'machine {machine.idx}: {press_count}')
            total_sum += press_count

    # enque termination marker
    large_machine_q.put(0)

    # print(f'waiting for {n_offloaded} large machines')
    
    with log_event_name_only(f'waiting for {n_offloaded} large machines'):
        while n_offloaded > 0:
            completed_machine = completed_machine_q.get()
            n_offloaded -= 1
            total_sum += completed_machine.best_soln
            # print(f'machine {completed_machine.idx} done, '
            #       f'soln = {completed_machine.best_soln}')

    for t in compute_threads:
        t.join()

    print(total_sum)

    
def main(args):
    filename = 'day10.in'
    n_threads = 3

    while len(args) > 0 and args[0][0] == '-':
        if args[0] == '-t':
            if len(args) < 2:
                print('No argument for -t option')
                return 1
            try:
                n_threads = int(args[1])
            except ValueError:
                print(f'Invalid thread count: {args[1]}')
                return 1
            del args[:2]
            
        else:
            print(f'Unrecognized argument: {args[0]}')
            return 1

    if len(args) > 1:
        print('Extra command line arguments (expecting a filename or nothing)')
        return 1
    
    if len(args) > 0:
        filename = args[0]
            
    # read input as a list of strings
    input = read_problem_input(filename)
    machines = [Machine.parse(line, idx) for idx, line in enumerate(input)]
  
    t0 = time.perf_counter_ns()
    # part1(machines)
    t1 = time.perf_counter_ns()
    # part2(machines)
    # part2_threaded(machines, n_threads)
    # part2_time_each_problem(machines, 10)
    # part2_single_problem(machines[23], 50)
    part2_threaded_large_ones(machines, n_threads)
    t2 = time.perf_counter_ns()
    # print(f'part1 {(t1-t0)/1e6:.2f} millis')
    print(f'part2 {(t2-t1)/1e6:.2f} millis')
    
    

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
