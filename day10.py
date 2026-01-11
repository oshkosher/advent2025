#!/usr/bin/env python3

"""
Advent of Code 2025, Day 10: Factory

Integer programming with boolean matrices

Use UV to run with threaded interpreter:
  uv run -p 3.14t python day10.py

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools
# import numpy as np
from math import floor, ceil

import threading
from queue import Queue

# linear programming library
# import pulp

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *

# import day10_solver


class Machine:
    def __init__(self, lights, buttons, button_bits, jolts):
        self.lights = lights

        # list of tuples, just like the input
        self.buttons = buttons

        # list of integers with the given bits set
        # i.e. (0,1,2,3) -> 15
        self.button_bits = button_bits
        
        self.jolts = jolts

    line_re = re.compile(r'\[([#.]*)\] (.*) {(.*)}')
        
    @staticmethod
    def parse(line):
        """
        Parse line like this:
          [.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
        into a Machine object.
        The first part [.##.] is a bit mask: 0110
        The second part '(3) (1,3), ...' is a bit mask for the part1
          problem, and a list of indices for part2.
        The third part {3,5,4,7} is a list of joltage values for part 3.
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

        return Machine(lights, button_ints, button_bits, jolts)


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
        # print(f'machine {i}')
        # print(f'  {m.lights}')
        # print(f'  {m.buttons!r}')
        # print(f'  {m.button_bits!r}')
        # print(f'  {m.jolts!r}')
        pressed_list = light_button_count_bfs(m)
        button_press_total += len(pressed_list)
        # print(f'  soln = {pressed_list!r}')

    # 434 is wrong (right for someone else)
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
    

def solve(original_matrix, b):
    
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

    # list free columns
    free_cols = find_free_var_columns(A)
    n_free = len(free_cols)

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


def joltage_presses(machine):

    # create a matrix representing what lights up when each
    # button is pressed
    A = Matrix(height = len(machine.jolts), width = len(machine.buttons))
    for c in range(A.width):
        button = machine.buttons[c]
        for r in button:
            A[r][c] = 1

    # find the best solution for the matrix
    return solve(A, machine.jolts)

        
def part2(machines):
    total_presses = 0
    for i, machine in enumerate(machines):
        total_presses += joltage_presses(machine)
    print(total_presses)


class ComputeThread(threading.Thread):
    def __init__(self, thread_id, todo_q, results_q):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.todo_q = todo_q
        self.results_q = results_q

    def run(self):
        getting_ns = 0
        putting_ns = 0
        jolting_ns = 0
        tasks_done = 0
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

            timer = time.perf_counter_ns()
            press_count = joltage_presses(machine)
            jolting_ns += time.perf_counter_ns() - timer
            
            tasks_done += 1
            timer = time.perf_counter_ns()
            self.results_q.put(press_count)
            putting_ns += time.perf_counter_ns() - timer

        getting_ms = f'{getting_ns / 1e6:.3f}'
        putting_ms = f'{putting_ns / 1e6:.3f}'
        jolting_ms = f'{jolting_ns / 1e6:.0f}'
        print(f'thread {self.thread_id}: {tasks_done} tasks, {jolting_ms}ms computing, {getting_ms}ms getting, {putting_ms}ms putting')
    

def part2_threaded(machines, n_compute_threads = 4):
    todo_q = Queue()
    results_q = Queue()

    compute_threads = []
    for i in range(n_compute_threads):
        t = ComputeThread(i, todo_q, results_q)
        t.start()
        compute_threads.append(t)

    nanos = time.perf_counter_ns()
    for machine in machines:
        todo_q.put(machine)
    todo_q.put(0)

    total_presses = 0
    for i in range(len(machines)):
        total_presses += results_q.get()
        # print(f'got {i}')
    nanos = time.perf_counter_ns() - nanos

    print(f'{total_presses} in {nanos / 1e6:.3f} ms')

    for t in compute_threads:
        t.join()


if __name__ == '__main__':
    thread_count = 4
    
    if sys.argv[1] == '-t' and len(sys.argv) >= 3:
        try:
            thread_count = int(sys.argv[2])
            if thread_count < 1:
                raise ValueError()
        except ValueError:
            print(f'Invalid thread count: {sys.argv[2]}')
            sys.exit(1)
        del sys.argv[1:3]

    # read input as a list of strings
    input = read_problem_input()
    machines = [Machine.parse(line) for line in input]
  
    t0 = time.perf_counter_ns()
    part1(machines)
    t1 = time.perf_counter_ns()
    part2(machines)
    # part2_threaded(machines, thread_count)
    t2 = time.perf_counter_ns()
    print(f'part1 {(t1-t0)/1e6:.2f} millis')
    print(f'part2 {(t2-t1)/1e6:.2f} millis')
