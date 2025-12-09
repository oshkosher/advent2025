#!/usr/bin/env python3

"""
Advent of Code 2025, Day 9: Movie Theater

Rectangular regions inside an irregular region

Ed Karrels, ed.karrels@gmail.com, December 2025
"""

# common standard libraries
import sys, os, re, itertools, math, collections, itertools, array
from PIL import Image, ImageDraw
from draw_grid import draw_grid

# change to the directory of the script so relative references work
from pathlib import Path
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))
from advent import *

"""
The input data is 496 points in the range 0..100000.
Building a dense grid to represent the whole thing would be infeasible.
100000 * 100000 = 10 billion. Even using a single byte for each pixel would
be 10 GB of memory. Perhaps the region can be defined in horizontal and vertical
strips.

Assuming the first coordinate is the row and the second is the column,
the tiles roughly define a circle, 100000 pixels wide, starting from
the bottom middle and progressing counter-clockwise. But there's a
channel cut out of the middle, like this:

.................................######.####...................................
.........................##########...#.#...#########...........................
.....................##.####..........#.#...........#####.......................
.................#####................#.#...............######..................
...............###....................#.#....................###................
............####......................#.#......................###..............
...........##.........................#.#........................###............
.........###..........................#.#..........................###..........
........##............................#.#............................###........
......###.............................#.#..............................##.......
.....##...............................#.#...............................##......
.....#................................#.#................................###....
....##................................#.#..................................#....
...##.................................#.#..................................##...
..##..................................#.#...................................##..
..#...................................#.#....................................#..
..#...................................#.#....................................##.
.##...................................#.#....................................##.
.##...................................#.#....................................##.
.#....................................#.#.....................................#.
.##...................................#.#.....................................#.
.##...................................#.#.....................................#.
.#....................................#.#....................................##.
.##...................................#.#....................................#..
..##..................................#.#...................................##..
...#..................................#.#...................................#...
...##.................................#.#...................................#...
....##................................#.#..................................##...
.....##...............................#.#................................###....
......##..............................#.#................................#......
.......##.............................#.#..............................###......
........###...........................#.#.............................##........
..........###.........................#.#...........................###.........
............###.......................#.#.........................###...........
..............###.....................#.#.......................###.............
................####..................#.#....................####...............
...................####...............#.#................#####..................
......................#####...........###............#####......................
..........................###########......###########..........................
....................................##########..................................

With the channel in the middle, you can't define a valid rectangle that spans
the center of the circle, because the rectangle must only use red and green
tiles, and the tiles in the channel are neither.

.................................######.####...................................
.........................##########...#.#...#########...........................
.....................##.####..........#.#...........#####.......................
.................#####................#.#...............######..................
...............###....................#.#....................###................
............####......................#.#......................###..............
...........##.........................#.#........................###............
.........###..........................#.#..........................###..........
........##............................#.#............................###........
......###OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO.##.......
.....##..OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO..##......
.....#...OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO...###....
....##...OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO.....#....
...##....OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO.....##...
..##.....OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO......##..
..#......OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO.......#..
..#......OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO.......##.
.##......OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO.......##.
.##......OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO.......##.
.#.......OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO........#.
.##......OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO........#.
.##......OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO........#.
.#.......OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO.......##.
.##......OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO.......#..
..##.....OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO......##..
...#.....OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO......#...
...##....OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO......#...
....##...OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO.....##...
.....##..OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO...###....
......##.OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO...#......
.......##OOOOOOOOOOOOOOOOOOOOOOOOOOOOO#.#OOOOOOOOOOOOOOOOOOOOOOOOOOOOO.###......
........###...........................#.#.............................##........
..........###.........................#.#...........................###.........
............###.......................#.#.........................###...........
..............###.....................#.#.......................###.............
................####..................#.#....................####...............
...................####...............#.#................#####..................
......................#####...........###............#####......................
..........................###########......###########..........................
....................................##########..................................

With the channel cutting off large rectangles in the middle, and each half of the
circle being concave, this greatly restricts which red tiles can be used as the
corners of the rectangle.

The channel doesn't quite reach the bottom, and its bottom corners
seem to be one of the few places a rectangle inside the region can be
found. I suspect every candidate rectangle uses one of the channel
corners, like this:

.................................######.####...................................
.........................##########...#.#...#########...........................
.....................##.####..........#.#...........#####.......................
.................#####.....OOOOOOOOOOO#.#...............######..................
...............###.........OOOOOOOOOOO#.#....................###................
............####...........OOOOOOOOOOO#.#......................###..............
...........##..............OOOOOOOOOOO#.#........................###............
.........###...............OOOOOOOOOOO#.#..........................###..........
........##.................OOOOOOOOOOO#.#............................###........
......###..................OOOOOOOOOOO#.#..............................##.......
.....##....................OOOOOOOOOOO#.#...............................##......
.....#.....................OOOOOOOOOOO#.#................................###....
....##.....................OOOOOOOOOOO#.#..................................#....
...##......................OOOOOOOOOOO#.#..................................##...
..##.......................OOOOOOOOOOO#.#...................................##..
..#........................OOOOOOOOOOO#.#....................................#..
..#........................OOOOOOOOOOO#.#....................................##.
.##........................OOOOOOOOOOO#.#....................................##.
.##........................OOOOOOOOOOO#.#....................................##.
.#.........................OOOOOOOOOOO#.#.....................................#.
.##........................OOOOOOOOOOO#.#.....................................#.
.##........................OOOOOOOOOOO#.#.....................................#.
.#.........................OOOOOOOOOOO#.#....................................##.
.##........................OOOOOOOOOOO#.#....................................#..
..##.......................OOOOOOOOOOO#.#...................................##..
...#.......................OOOOOOOOOOO#.#...................................#...
...##......................OOOOOOOOOOO#.#...................................#...
....##.....................OOOOOOOOOOO#.#..................................##...
.....##....................OOOOOOOOOOO#.#................................###....
......##...................OOOOOOOOOOO#.#................................#......
.......##..................OOOOOOOOOOO#.#..............................###......
........###................OOOOOOOOOOO#.#.............................##........
..........###..............OOOOOOOOOOO#.#...........................###.........
............###............OOOOOOOOOOO#.#.........................###...........
..............###..........OOOOOOOOOOO#.#.......................###.............
................####.......OOOOOOOOOOO#.#....................####...............
...................####....OOOOOOOOOOO#.#................#####..................
......................#####OOOOOOOOOOO###............#####......................
..........................###########......###########..........................
....................................##########..................................

So we only need to test both channel corners against the red tiles in the upper
quadrant above them. 
"""


def parse_input(input):
    return [tuple([int(n) for n in line.split(',')]) for line in input]


def area(t1, t2):
    return (abs(t1[0]-t2[0])+1) * (abs(t1[1]-t2[1])+1)


def part1(tiles):
    max_area = 0
    for i in range(len(tiles)-1):
        for j in range(i+1, len(tiles)):
            max_area = max(max_area, area(tiles[i], tiles[j]))
    print(max_area)


def find_right_channel_tile(tiles):
    """
    Returns the index of the right channel tile.
    
    The tiles define a circle, starting from the bottom and going counter-clockwise.
    The two tiles defining the channel appear in the middle of the data.
    This returns the index of the channel tile on the right. The next tile is the
    left channel tile.
    """
    for i, tile in enumerate(tiles):
        prev = tiles[i-1]
        if tile[0] - prev[0] > 5000:
            return i
    return None


def find_column_ranges(tiles, channel_right_idx):
    """
    For each column of the grid, compute the minimum and maximum rows such that
    the midpoint is included in the range, and every point between the minimum and
    maximum rows is inside the region.

    Returns two arrays min[] and max[]. For a given column c, all cells from
    min[c] through max[c] are within the region (red or green tiles).

    channel_right_idx is the index of the southeast channel tile.
    """

    channel_bottom = tiles[channel_right_idx][0]
    channel_left = tiles[channel_right_idx+1][1]
    channel_right = tiles[channel_right_idx][1]

    # print((channel_bottom, channel_left, channel_right))

    def filler(value, count):
        for i in range(count):
            yield value

    # the arrays are large, so use efficient storage
    col_min = array.array('i', filler(0, 100000))
    col_max = array.array('i', filler(-1, 100000))

    # start in the southeast quadrant
    ti = 0  # tile index
    half = 50000
    ci = half  # column index
    
    while tiles[ti][0] > half:
        break
    
    return col_min, col_max


def northwest_enclosed(tiles, channel_right, corner_tile_idx):
    # top row of the region
    top = tiles[corner_tile_idx][0]

    # left column of the region
    left = tiles[corner_tile_idx][1]

    for ti in range(channel_right+3, len(tiles)):
        tile = tiles[ti]

        # skip tiles above the region
        if tile[0] < top:
            continue
        
        if tile[0] > top and tile[1] > left:
            return False

        # no need to check past the left edge of the region
        if tile[1] < left: break
        
    return True

    
def southwest_enclosed(tiles, channel_right, corner_tile_idx):

    # bottom row of the region
    bottom = tiles[channel_right+1][0]

    # left column of the region
    left = tiles[corner_tile_idx][1]
    
    # move clockwise from the southernmost point
    for ti in range(len(tiles)-1, 0, -1):
        tile = tiles[ti]

        # skip tiles below the region
        if tile[0] > bottom:
            continue

        if tile[0] < bottom and tile[1] > left:
            return False

        # no need to check past the left edge of the region
        if tile[1] < left:
            break

    return True


def northeast_enclosed(tiles, channel_right, corner_tile_idx):
    # top row of the region
    top = tiles[corner_tile_idx][0]

    # right column of the region
    right = tiles[corner_tile_idx][1]

    for ti in range(channel_right-2, 0, -1):
        tile = tiles[ti]

        # skip tiles above the region
        if tile[0] < top:
            continue
        
        if tile[0] > top and tile[1] < right:
            return False

        # no need to check past the right edge of the region
        if tile[1] > right: break
        
    return True

    
def southeast_enclosed(tiles, channel_right, corner_tile_idx):

    # bottom row of the region
    bottom = tiles[channel_right][0]

    # right column of the region
    right = tiles[corner_tile_idx][1]
    
    # move counter-clockwise from the southernmost point
    for ti in range(0, channel_right):
        tile = tiles[ti]

        # skip tiles below the region
        if tile[0] > bottom:
            continue

        if tile[0] < bottom and tile[1] < right:
            return False

        # no need to check past the right edge of the region
        if tile[1] > right:
            break

    return True
        
    

def part2(tiles, image_draw):
    channel_right = find_right_channel_tile(tiles)
    # print(tiles[channel_right])
    # print(tiles[channel_right+1])

    # column_ranges = find_column_ranges(tiles, channel_right)

    best_area = 0
    best_pair = None

    # find the largest rectangle defined by the left channel tile and a tile in the
    # northwest quadrant
    # ti = channel_right + 3
    channel_tile = tiles[channel_right+1]
    for ti in range(channel_right+3, len(tiles)):
        if tiles[ti][0] >= 50000:
            break

        if (northwest_enclosed(tiles, channel_right, ti)
            and southwest_enclosed(tiles, channel_right, ti)):

            if image_draw:
                x1 = tiles[ti][1] / 100
                y1 = tiles[ti][0] / 100
                x2 = channel_tile[1] / 100
                y2 = channel_tile[0] / 100
                image_draw.rectangle([x1, y1, x2, y2],
                                     fill = None, outline = (0, 255, 0))
            
            rect_area = area(channel_tile, tiles[ti])

            # print(f'{tiles[ti]}: {rect_area}')
            
            if rect_area > best_area:
                best_area = rect_area
                best_pair = (channel_tile, tiles[ti])

    # print('now check northeast')
                
    # find the largest rectangle defined by the right channel tile and a tile in the
    # northeast quadrant
    channel_tile = tiles[channel_right]

    for ti in range(channel_right - 2, 0, -1):
        tile = tiles[ti]

        # stop when eastern edge of the circle is reached
        if tile[0] >= 50000:
            break

        if (northeast_enclosed(tiles, channel_right, ti)
            and southeast_enclosed(tiles, channel_right, ti)):

            if image_draw:
                x1 = channel_tile[1] / 100
                y1 = tiles[ti][0] / 100
                x2 = tiles[ti][1] / 100
                y2 = channel_tile[0] / 100
                image_draw.rectangle([x1, y1, x2, y2],
                                     fill = None, outline = (0, 255, 0))
            
            rect_area = area(channel_tile, tiles[ti])

            # print(f'{tiles[ti]}: {rect_area}')
            
            if rect_area > best_area:
                best_area = rect_area
                best_pair = (channel_tile, tiles[ti])
        
    # print(best_pair)
    print(best_area)


def draw_image(tiles):
    """
    The original tile coordinates are in the range 0..100000
    Scale them down by a factor of 100
    """
    filename = 'day9.png'
    scale = 100
    width = height = 1000
    grid = grid_create(height, width, True)
    # for r,c in tiles:
    #     grid[r//scale][c//scale] = 'r'
    # draw_grid(grid, 'day9.png', cell_size=4)

    bg_color = (255, 255, 255)
    cell_color = (255, 0, 0)
    im = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(im)

    def scl(tile):
        return tile[1]//scale, tile[0]//scale

    for ti in range(len(tiles)):
        pi = ti - 1 if ti > 0 else len(tiles)-1
        draw.line((scl(tiles[pi]), scl(tiles[ti])), cell_color)
        # draw.rectangle((x, y, x+1, y+1), cell_color)

    # draw center lines
    # draw.line([(0, 500), (1000,500)], (0, 0, 255))
    # draw.line([(500, 0), (500,1000)], (0, 0, 255))

    # fill in channel
    channel_right = find_right_channel_tile(tiles)
    corners = [tiles[channel_right+2][1] // scale,
               tiles[channel_right+2][0] // scale,
               tiles[channel_right][1] // scale,
               tiles[channel_right][0] // scale]
    draw.rectangle(corners, (0,0,0))

    # im.save(filename)
    # print(filename + ' written')
    return im, draw


def draw_text_image(tiles):
    grid = grid_create(80, 80, True)
    scale = 80 / 100000
    for r, c in tiles:
        grid[int(r*scale*.5)][int(c*scale)] = '#'
    grid_print(grid)


if __name__ == '__main__':
    # read input as a list of strings
    input = read_problem_input()
    tiles = parse_input(input)

    image, image_draw = None, None
    
    # image, image_draw = draw_image(tiles)
    # draw_text_image(tiles)

    part1(tiles)
    part2(tiles, image_draw)

    if image:
        filename = 'day9.png'
        image.save(filename)
        print(filename + ' written')
