#!/usr/bin/env python3

import sys
# install this as pillow: "python -m pip install Pillow"
from PIL import Image, ImageDraw
from advent import read_grid


defaut_cell_color_map = {
  '.': None,
  '#': (0,0,0),
  'S': (255,0,0),
  'O': (0,0,255),
  'o': (150,150,255),
  'r': (255,0,0),
  'y': (255,255,0),
  'g': (0,255,0),
  'c': (0,255,255),
  'b': (0,0,255),
  'm': (255,0,255),
  'k': (0,0,0),
  'w': (255,255,255),
  '1': (0, 200, 0),
  }


def draw_grid(grid,
              output_filename,
              cell_color_map = None,
              cell_size = 20,
              bg_color = (255,255,255),
              border_color = (200,200,200),
              inset = 2,
              ):

  n_rows = len(grid)
  n_cols = len(grid[0])
  width_px = n_cols * cell_size + 1
  height_px = n_rows * cell_size + 1

  im = Image.new('RGB', (width_px,height_px), bg_color)
  draw = ImageDraw.Draw(im)

  user_cell_color_map = cell_color_map
  cell_color_map = defaut_cell_color_map.copy()
  if user_cell_color_map:
    for k,v in user_cell_color_map.items():
      cell_color_map[k] = v

  for r in range(0, n_rows + 1):
    draw.line(((0, r * cell_size), (width_px, r * cell_size)), border_color)
    
  for c in range(0, n_cols + 1):
    draw.line(((c * cell_size, 0), (c * cell_size, height_px)), border_color)

  for r in range(n_rows):
    for c in range(n_cols):
      x = grid[r][c]
      color = cell_color_map.get(x, None)
      if color:
        x = c * cell_size + inset
        x2 = (c+1) * cell_size - inset
        y = r * cell_size + inset
        y2 = (r+1) * cell_size - inset
        draw.rectangle([x, y, x2, y2], color)
        
    
  im.save(output_filename)
  print(f'Wrote {n_rows}x{n_cols} grid to {output_filename}')


# grid = read_grid(open('day21.in.txt'))
# draw_grid(grid, 'draw_grid.png')

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print('\n  draw_grid.py <in> <out.png>\n')
    sys.exit(1)

  (in_fn, out_fn) = sys.argv[1:3]
  with open(in_fn) as inf:
    grid = read_grid(inf)
  draw_grid(grid, out_fn)
