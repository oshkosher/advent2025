import java.util.List;
import java.util.ArrayList;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Point;
import java.awt.Graphics;
import java.awt.Graphics2D;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.WindowConstants;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


public class day9_visual extends JComponent {

  public static void main(String[] args) {
    String filename = "day9.in";
    
    if (args.length > 1)
      filename = args[1];

    ArrayList<Point> red_tiles;
    try {
      red_tiles = readInputFile(filename);
    } catch (IOException e) {
      System.out.println("Failed to read " + filename);
      return;
    }

    day9_visual viz = new day9_visual(red_tiles);

    JFrame frame = new JFrame("Day 9 Visualization");
    frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    frame.add(viz, "Center");
    frame.setSize(new Dimension(1000, 1000));
    frame.setLocationRelativeTo(null);
    frame.setVisible(true);

  }


  static ArrayList<Point> readInputFile(String filename) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(filename));

    ArrayList<Point> tiles = new ArrayList<Point>();
    
    String line;
    while ((line = reader.readLine()) != null) {

      // parse line in the form "12345,67890"
      int comma = line.indexOf(',');
      assert comma > 0;
      Point p = new Point();
      p.x = Integer.parseInt(line, 0, comma, 10);
      p.y = Integer.parseInt(line, comma+1, line.length(), 10);
      tiles.add(p);
    }

    return tiles;
  }

  

  public record MinMax(Point min, Point max) {}

  
  static MinMax pointRange(List<Point> points) {
    Point min = new Point(Integer.MAX_VALUE, Integer.MAX_VALUE);
    Point max = new Point(Integer.MIN_VALUE, Integer.MIN_VALUE);

    for (Point p : points) {
      if (p.x < min.x) min.x = p.x;
      if (p.y < min.y) min.y = p.y;
      if (p.x > max.x) max.x = p.x;
      if (p.y > max.y) max.y = p.y;
    }
    
    return new MinMax(min, max);
  }


  private Color foreground_color = Color.BLACK;
  private Color background_color = Color.WHITE;
  private Point min, max;
  private int x_points[], y_points[];

  
  public day9_visual(ArrayList<Point> tiles) {
    MinMax range = pointRange(tiles);
    min = range.min;
    max = range.max;

    setBackground(background_color);
    
    // copy the input data into arrays compatible with Graphics.fillPolygon()
    x_points = new int[tiles.size()];
    y_points = new int[tiles.size()];

    for (int i=0; i < tiles.size(); ++i) {
      x_points[i] = tiles.get(i).x;
      y_points[i] = tiles.get(i).y;
    }
  }

  
  /* set scale and offset so all the input points fit in the window */
  private void rescale(Graphics2D g) {
    Dimension size = getSize();
    
    double x_scale = (double) size.width / (max.x - min.x);
    double y_scale = (double) size.height / (max.y - min.y);
    double scale = Math.min(x_scale, y_scale);

    g.scale(scale, scale);
    g.translate(-min.x, -min.y);
  }
  
  public void paintComponent(Graphics g1) {
    Graphics2D g = (Graphics2D) g1;
    rescale(g);

    g.fillPolygon(x_points, y_points, x_points.length);
  }
    
}
