import sys
from collections import deque


class Node():
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action


class StackFrontier():
    def __init__(self):
        self.frontier = deque()

    def add(self, node):
        self.extra_per_node(node)
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self, **kwargs):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier.pop()
            return node

    @staticmethod
    def extra_per_node(node):
        return  # do nothing for this


class QueueFrontier(StackFrontier):
    def remove(self, **kwargs):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier.popleft()
            return node


class GreedyFrontier(StackFrontier):
    @staticmethod
    def heuristic(node, maze) -> int:
        # Manhattan distance heuristic
        s1 = node.state
        return abs(s1[0]-maze.goal[0]) + abs(s1[1]-maze.goal[1])

    def __init__(self):
        self.frontier = []

    def remove(self, maze):
        if self.empty():
            raise Exception("empty frontier")
        else:
            index = self.frontier.index(
                min(self.frontier, key=lambda Node: self.heuristic(Node, maze)))
            return self.frontier.pop(index)


class A_starFrontier(GreedyFrontier):
    man_hattan = GreedyFrontier.heuristic

    @staticmethod
    def heuristic(node, maze) -> int:
        return A_starFrontier.man_hattan(node, maze)+node.walked

    @staticmethod
    def extra_per_node(node):
        if not node.parent:
            node.walked = 0
        else:
            node.walked = node.parent.walked+1


class Maze():
    def __init__(self, filename):
        # Read file and set height and width of maze
        with open(filename) as f:
            contents = f.read()
        # Validate start and goal
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")
        # Determine height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)
        # Keep track of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)
        self.solution = None

    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()

    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]
        # result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                yield ((action, (r, c)))

    def solve(self, algo_frontier=StackFrontier):
        """Finds a solution to maze, if one exists."""
        # Keep track of number of states explored
        self.num_explored = 0
        # Initialize frontier to just the starting position
        start = Node(state=self.start, parent=None, action=None)
        algo_frontier.extra_per_node(start)  # for some specific algors
        frontier = algo_frontier()
        frontier.add(start)
        # Initialize an empty explored set
        self.explored = set()
        # Keep looping until solution found
        while True:
            # If nothing left in frontier, then no path
            if frontier.empty():
                raise Exception("no solution")
            # Choose a node from the frontier
            node = frontier.remove(maze=self)
            self.num_explored += 1
            # If node is the goal, then we have a solution
            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return
            # Mark node as explored
            self.explored.add(node.state)
            # Add neighbors to frontier
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    frontier.add(child)

    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2
        # Create a blank canvas
        img = Image.new("RGBA", (self.width * cell_size,
                        self.height * cell_size), "black")
        draw = ImageDraw.Draw(img)
        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                # Walls
                if col:
                    fill = (40, 40, 40)
                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)
                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)
                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)
                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)
                # Empty cell
                else:
                    fill = (237, 240, 252)
                # Draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )
        img.save(filename)


def cmd_prompt():
    global args
    import argparse
    parser = argparse.ArgumentParser(description="CS50 maze program.")
    parser.epilog = """- Argparse support added.
        - Greedy Best-First and A* algorithms implemented.
        - by @IAMAJAYPRO"""
    parser.add_argument("file", help="Input maze file (txt).")
    ftier_gp = parser.add_mutually_exclusive_group()  # frontier group
    ftier_gp.add_argument("-f", "--frontier", type=str, help="Frontier to be used, default: DFS",
                          choices=["stack", "queue", "greedy"], default="stack")
    ftier_gp.add_argument("-q", "--queue", "--bfs", action='store_const', const="queue", dest="frontier",
                          help="Use queue as the frontier(Breadth First Search).")
    ftier_gp.add_argument("-g", "--greedy", action='store_const', const="greedy", dest="frontier",
                          help="Use Greedy Best-First Search.")
    ftier_gp.add_argument("-A", "--A*", "--AStar", action='store_const', const="a_star", dest="frontier",
                          help="Use AStar Search. (h(n) + g(n))")
    parser.add_argument("-e", "--explored", action='store_true',
                        help="Shows the explored states in output")
    args = parser.parse_args()
    # print(args)


def main():
    cmd_prompt()  # args is global variable auto set
    frontiers = dict(stack=StackFrontier, queue=QueueFrontier,
                     greedy=GreedyFrontier, a_star=A_starFrontier)
    m = Maze(args.file)
    print("Maze:")
    m.print()
    print("Solving...")
    m.solve(algo_frontier=frontiers[args.frontier])
    print("States Explored:", m.num_explored)
    print("Solution:")
    m.print()
    m.output_image("maze.png", show_explored=args.explored)


if __name__ == "__main__":
    main()
