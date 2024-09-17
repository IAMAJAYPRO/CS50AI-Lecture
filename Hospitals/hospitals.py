import random

CELL_SIZE, CELL_SIZE = (None,)*2


class Space():

    def __init__(self, height, width, num_hospitals):
        """Create a new state space with given dimensions."""
        self.height = height
        self.width = width
        self.num_hospitals = num_hospitals
        self.houses = set()
        self.hospitals = set()
        self.best_hospitals = None

    def add_house(self, row, col):
        """Add a house at a particular location in state space."""
        self.houses.add((row, col))

    def available_spaces(self):
        """Returns all cells not currently used by a house or hospital."""

        # Consider all possible cells
        candidates = set(
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
        )

        # Remove all houses and hospitals
        for house in self.houses:
            candidates.remove(house)
        for hospital in self.hospitals:
            candidates.remove(hospital)
        return candidates

    def hill_climb(self, maximum=None, image_prefix=None, log=False):
        """Performs hill-climbing to find a solution."""
        count = 0

        # Start by initializing hospitals randomly
        self.hospitals = set()
        for i in range(self.num_hospitals):
            self.hospitals.add(random.choice(list(self.available_spaces())))
        if log:
            print("Initial state: cost", self.get_cost(self.hospitals))
        if image_prefix:
            self.output_image(f"{image_prefix}{str(count).zfill(3)}.png")

        # Continue until we reach maximum number of iterations
        while maximum is None or count < maximum:
            count += 1
            best_neighbors = []
            best_neighbor_cost = None

            # Consider all hospitals to move
            for hospital in self.hospitals:

                # Consider all neighbors for that hospital
                for replacement in self.get_neighbors(*hospital):

                    # Generate a neighboring set of hospitals
                    neighbor = self.hospitals.copy()
                    neighbor.remove(hospital)
                    neighbor.add(replacement)

                    # Check if neighbor is best so far
                    cost = self.get_cost(neighbor)
                    if best_neighbor_cost is None or cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbors = [neighbor]
                    elif best_neighbor_cost == cost:
                        best_neighbors.append(neighbor)

            # None of the neighbors are better than the current state
            if best_neighbor_cost >= self.get_cost(self.hospitals):
                return self.hospitals

            # Move to a highest-valued neighbor
            else:
                if log:
                    print(f"Found better neighbor: cost {best_neighbor_cost}")
                self.best_hospitals = self.hospitals
                self.hospitals = random.choice(best_neighbors)

            # Generate image
            if image_prefix:
                self.output_image(f"{image_prefix}{str(count).zfill(3)}.png")

    def random_restart(self, maximum, image_prefix=None, log=False):
        """Repeats hill-climbing multiple times."""
        best_hospitals = None
        best_cost = None

        # Repeat hill-climbing a fixed number of times
        for i in range(maximum):
            hospitals = self.hill_climb()
            cost = self.get_cost(hospitals)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_hospitals = hospitals
                if log:
                    print(f"{i}: Found new best state: cost {cost}")
            else:
                if log:
                    print(f"{i}: Found state: cost {cost}")

            if image_prefix:
                self.output_image(f"{image_prefix}{str(i).zfill(3)}.png")

        return best_hospitals

    def get_cost(self, hospitals):
        """Calculates sum of distances from houses to nearest hospital."""
        cost = 0
        for house in self.houses:
            cost += min(
                abs(house[0] - hospital[0]) + abs(house[1] - hospital[1])
                for hospital in hospitals
            )
        return cost

    def get_neighbors(self, row, col):
        """Returns neighbors not already containing a house or hospital."""
        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]
        neighbors = []
        for r, c in candidates:
            if (r, c) in self.houses or (r, c) in self.hospitals:
                continue
            if 0 <= r < self.height and 0 <= c < self.width:
                neighbors.append((r, c))
        return neighbors

    def output_image(self, filename, prefix_folder=True, best_hospitals=False):
        """Generates image with all houses and hospitals."""

        if prefix_folder:
            filename = "output/"+filename
        from PIL import Image, ImageDraw, ImageFont
        cell_size = CELL_SIZE
        cell_border = CELL_BORDER
        cost_size = 40
        padding = 10

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size,
             self.height * cell_size + cost_size + padding * 2),
            "white"
        )
        house = Image.open("assets/images/House.png").resize(
            (cell_size, cell_size)
        )
        hospital = Image.open("assets/images/Hospital.png").resize(
            (cell_size, cell_size)
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 30)
        draw = ImageDraw.Draw(img)
        "bit.ly/encrypted_credits"

        hospitals = self.hospitals if not best_hospitals else self.best_hospitals
        for i in range(self.height):
            for j in range(self.width):

                # Draw cell
                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                draw.rectangle(rect, fill="black")

                if (i, j) in self.houses:
                    img.paste(house, rect[0], house)
                if (i, j) in hospitals:
                    img.paste(hospital, rect[0], hospital)

        # Add cost
        draw.rectangle(
            (0, self.height * cell_size, self.width * cell_size,
             self.height * cell_size + cost_size + padding * 2),
            "black"
        )
        draw.text(
            (padding, self.height * cell_size + padding),
            f"Cost: {self.get_cost(hospitals)}",
            fill="white",
            font=font
        )

        img.save(filename)


# Create a new space and add houses randomly
def cmd_prompt():
    import argparse
    parser = argparse.ArgumentParser(
        description="CS50 AI program for Hospitals")
    parser.epilog = "Modified by @iamajaypro"
    parser.add_argument('--size', nargs=2, default=[10, 20], type=int, metavar=('HEIGHT', 'WIDTH'),
                        help='specify size as height and width')
    parser.add_argument('--houses', type=int, default=15,
                        help='Specify the number of houses')
    parser.add_argument('--hospitals', type=int, default=3,
                        help='Specify the number of hospitals')
    parser.add_argument('--last_only', action='store_true',
                        help='output only the last stage image')
    parser.add_argument('--cellsize', type=int, default=20,
                        help='Specify the cell size')
    parser.add_argument("--algo", "--algorithm", default="hill",
                        choices=["hill", "random"], help="random: random restart")

    args = parser.parse_args()
    args.HEIGHT, args.WIDTH = args.size

    global CELL_SIZE, CELL_BORDER
    CELL_SIZE = args.cellsize
    CELL_BORDER = max(CELL_SIZE//20, 1)
    return args


def main(args):
    s = Space(height=args.HEIGHT, width=args.WIDTH,
              num_hospitals=args.hospitals)
    for i in range(args.houses):
        s.add_house(random.randrange(s.height), random.randrange(s.width))

    # Use local search to determine hospital placement
    parameters = dict(
        image_prefix=None if args.last_only else "hospitals", log=True)
    match args.algo.lower():
        case "hill":
            hospitals = s.hill_climb(**parameters)
        case "random":
            hospitals = s.random_restart(**parameters, maximum=10)

    if args.last_only:
        s.output_image("Solution.png", best_hospitals=True)


if __name__ == "__main__":
    args = cmd_prompt()
    main(args)
