#!/usr/bin/env python3

"""
Advent of Code 2025, Day 10: Factory

Integer programming with boolean matrices

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools
import numpy as np

# linear programming library
import pulp

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *


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


def joltage_presses(machine):
    """
    1*(3)
    3*(1,3)
    0*(2)
    3*(2,3)
    1*(0,2)
    2*(0,1)

    0: 3
    1: 5
    2: 4
    3: 7

    This problem is finding a vector x such that A x = b in an
    underconstrained system, looking for a minimum sum of the elements
    in vector x.
    Also:
      - the input matrix is all 0's and 1's.
      
    
    [0 0 0 0 1 1  * [1 = [3
     0 1 0 0 0 1     3    5
     0 0 1 1 1 0     0    4
     1 1 0 1 0 0]    3    7]
                     1    
                     2]

    Matrix
      each column is the bits in a button
      one column per button
    goal = jolts vector
    solution vector = number of times each buttons should be pressed

    1 * 1000
   +3 * 1010
   +0 *  100
   +3 * 1100
   +1 *  101
   +2 *   11

    FrobeniusSolve[{1000, 1010, 100, 1100, 101, 11}, 7453]
      finds the solution, along with 4000 others
    
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

    print(f'problem_list.append( (\n{A_ints!r},\n  {machine.jolts!r},\n  {soln!r}\n))')
    
    # soln_vector = np.array(soln)
    # print(f'A x = {A @ soln_vector}')
    # check_float = [v for v in A @ soln_vector]
    # check = [round(v) for v in check_float]
    # if check != machine.jolts:
    #     print(f'ERROR, goal={machine.jolts}, result={check} or {check_float}')
    # print(repr(check))

    return sum(soln)


def joltage_output_math(machine):
    """
    Just output the math problems so I can use them as direct sample input
    while working on my own solver.

    Output the boolean matrix A and goal vector b, and whether
    the problem is overconstrained, underconstrained, or conventionally
    solvable.
    
    [0 0 0 0 1 1  * [1 = [3
     0 1 0 0 0 1     3    5
     0 0 1 1 1 0     0    4
     1 1 0 1 0 0]    3    7]
                     1    
                     2]

    augmented matrix:
      0 0 0 0 1 1 3
      0 1 0 0 0 1 5
      0 0 1 1 1 0 4
      1 1 0 1 0 0 7

    row-reduced:
    1 0 0 1 0 -1 2
    0 1 0 0 0  1 5
    0 0 1 1 0 -1 1
    0 0 0 0 1  1 3

    meaning:
      x0 + x3 - x5 = 2
      x1 + x5 = 5
      x2 + x3 - x5 = 1
      x4 + x5 = 3

    solved for each variable independently:
      x0 = 2 - x3 + x5
      x1 = 5 - x5
      x2 = 1 - x3 + x5
      x3 = 1 - x2 + x5
      x4 = 3 - x5
      x5 = 5 - x1

      Starting from x1, its possible values are 0..5
        scanning value: x1
      Then x5 = 5 - x1
        compuated value: x1
      And x4 = 3 - x5 = 3 - (5 - x1) = x1 - 2
        computed value: x4
      x0 + x3 = x5 + 2
        possible values x0 = 0 .. x5+2
        scanning value: x0
        x3 = 0 .. x5 + 2 - x0
        computed value: x3
      x2 = 1 - x3 + x5
        computed value: x2

    Or based on the original matrix:

    [0 0 0 0 1 1   = [3
     0 1 0 0 0 1      5
     0 0 1 1 1 0      4
     1 1 0 1 0 0]     7]

     x4 + x5 = 3
     x1 + x5 = 5
     x2 + x3 + x4 = 4
     x0 + x1 + x3 = 7
    
    Let x1 = 3. That simplified two formulas:

    Apply this to augmented matrix by adding a row:
    
     0 0 0 0 1 1 3
     0 1 0 0 0 1 5
     0 0 1 1 1 0 4
     1 1 0 1 0 0 7
     0 1 0 0 0 0 3

    Simplify the matrix by subtracting that row from everyone with an x1:

     0 0 0 0 1 1 3
     0 0 0 0 0 1 2
     0 0 1 1 1 0 4
     1 0 0 1 0 0 4
     0 1 0 0 0 0 3

     x4 + x5 = 3
     x5 = 2
     x2 + x3 + x4 = 4
     x0 + x3 = 4
    
    x5 is now solved, subtract that from other rows:
    
     0 0 0 0 1 0 1
     0 0 0 0 0 1 2
     0 0 1 1 1 0 4
     1 0 0 1 0 0 4
     0 1 0 0 0 0 3

     x4 = 1
     x2 + x3 + x4 = 4
     x0 + x3 = 4

    x4 solved, use it:
    
     0 0 0 0 1 0 1
     0 0 0 0 0 1 2
     0 0 1 1 0 0 3
     1 0 0 1 0 0 4
     0 1 0 0 0 0 3

    When guessing a value, choose a value in a formula with a minimum RHS.
     x2 + x3 = 3
     x0 + x3 = 4

    Sample problem 2

      1 0 1 1 0  7
      0 0 0 1 1  5
      1 1 0 1 1 12
      1 1 0 0 1  7
      1 0 1 0 1  2

    row-reduced

      1  0  1  0  0  2
      0  1 -1  0  0  5
      0  0  0  1  0  5
      0  0  0  0  1  0
      0  0  0  0  0  0

      x0 + x2 = 2
      x1 - x2 = 5  or x1 = 5 + x2
      x3 = 5
      x4 = 0

      try x0 = 0 (remove zero rows, add a row for new value)

      1  0  1  0  0  2
      0  1 -1  0  0  5
      0  0  0  1  0  5
      0  0  0  0  1  0
      1  0  0  0  0  0
        reduce
      1  0  0  0  0  0
      0  1  0  0  0  7
      0  0  1  0  0  2
      0  0  0  1  0  5
      0  0  0  0  1  0
    

      try x0 = 1
        x2 = 1
        x1 = 6
      try x0 = 2

      1  0  1  0  0  2
      0  1 -1  0  0  5
      0  0  0  1  0  5
      0  0  0  0  1  0
      1  0  0  0  0  2
        reduce
      1  0  0  0  0  2
      0  1  0  0  0  5
      0  0  1  0  0  0
      0  0  0  1  0  5
      0  0  0  0  1  0

        preferred, for minimum total

      soln = [2, 5, 0, 5, 0]


    Sample problem 3
      1 1 1 0 10
      1 0 1 1 11
      1 0 1 1 11
      1 1 0 0  5
      1 1 1 0 10
      0 0 1 0  5

    row-reduced
      1  0  0  1  6
      0  1  0 -1 -1
      0  0  1  0  5
      0  0  0  0  0
      0  0  0  0  0
      0  0  0  0  0

      x0 + x3 = 6
      x1 - x3 = -1
      x2 = 5

    After row-reduction, the leading entry in every nonempty row will be 1.
    Since this started with a boolean/binary matrix, will all the remaining
    entries (except the last column) be 1, 0, or -1?


    

    """

    height = len(machine.jolts)
    width = len(machine.buttons)

    A = [[0] * width for _ in range(height)]

    for c in range(width):
        button = machine.buttons[c]
        for r in button:
            A[r][c] = 1

    goal = np.array(machine.jolts)

    print('A = {')
    for row in A:
        print('  {' + ', '.join([str(x) for x in row]) + '},')
    print('}')
    print('b = ' + repr(machine.jolts))
    if width > height:
        print('# solution: underconstrained')
    elif width < height:
        print('# solution: overconstrained')
    else:
        print('# solution: easy')
    
        
def part2(machines):
    print('problem_list = []')
    
    total_presses = 0
    for i, machine in enumerate(machines):
        print(f'# machine {i}')
        total_presses += joltage_presses(machine)
    # print(total_presses)
    
        
def part2_list_problems(machines):
    for i, machine in enumerate(machines):
        print(f'# machine {i}')
        joltage_output_math(machine)
        print()


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()
    machines = [Machine.parse(line) for line in input]
  
    t0 = time.perf_counter_ns()
    # part1(machines)
    t1 = time.perf_counter_ns()
    part2(machines)
    # part2_list_problems(machines)
    t2 = time.perf_counter_ns()
    # print(f'part1 {(t1-t0)/1e6:.2f} millis')
    # print(f'part2 {(t2-t1)/1e6:.2f} millis')
