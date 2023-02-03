from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

OPEN = 0
OBSTACLE = 1
ROBOT = 2

Direction = Tuple[int, int]
U: Direction = (0, 1)
UR: Direction = (1, 1)
R: Direction = (1, 0)
DR: Direction = (1, -1)
D: Direction = (0, -1)
DL: Direction = (-1, -1)
L: Direction = (-1, 0)
UL: Direction = (-1, 1)
directions = [U, UR, R, DR, D, DL, L, UL]

GOAL = 100.0
OBSTACLE_PENALTY = -50.0
MOVEMENT_PENALTY = -1.0


class Map:
    def __init__(self, goal: Tuple[int, int] = (10, 7), robot=(40, 7)):
        self.goal = goal
        self.robot = robot
        self.__init_map__()

    def __init_map__(self):
        self.map: List[List[int]] = []

        with open("./week4/programming/map.dat", "r") as file:
            lines = file.readlines()

            for index, line in enumerate(lines):
                self.map.append([])
                for char in line:
                    if char == "\n":
                        continue
                    self.map[index].append(int(char))

    def get(self, xy: Tuple[int, int]) -> int:
        return self.map[xy[1]][xy[0]]

    def shape(self) -> Tuple[int, int]:
        return (len(self.map), len(self.map[0]))

    def print(self):
        for row in self.map:
            for col in row:
                print(col, end="")
            print()

    def print_value_map(self):
        for y in range(len(self.map)):
            for x in range(len(self.map[0])):
                xy = (x, y)
                if self.is_obstacle(xy):
                    print("---", end=" ")
                elif self.goal == xy:
                    print(self.values[xy], end=" ")
                else:
                    print(round(self.values[xy], 1), end=" ")
            print()

    def draw(self, pixels_per=10):
        h, w = self.shape()
        img = Image.new("RGB", (w * pixels_per, h * pixels_per))
        draw = ImageDraw.Draw(img)

        for r, row in enumerate(self.map):
            for c, cell in enumerate(row):
                if (c, r) == self.goal:
                    color = "red"
                elif (c, r) == self.robot:
                    color = "green"
                elif cell == OPEN:
                    color = "white"
                elif cell == OBSTACLE:
                    color = "black"

                upper_left = (c * pixels_per, r * pixels_per)
                bottom_right = (upper_left[0] + pixels_per, upper_left[1] + pixels_per)
                draw.rectangle([upper_left, bottom_right], color)

        return img

    def set_robot(self, xy):
        self.robot = xy

    def move_robot(self, direction: Direction):
        xy = tuple(np.add(np.array(self.robot), np.array(direction)))
        if self.get(xy) == 1:
            raise Exception("Illegal move")
        else:
            self.set_robot(xy)

    def get_actions(self, xy: Tuple[int, int]) -> List[Direction]:
        """
        get_actions takes a given xy coordinate, then returns the
        possible open actions available in the order relative to
        xy clockwise:
        812
        7#3
        654
        """
        actions = []
        for direction in directions:
            xy = self.robot
            new_xy = tuple(np.add(np.array(xy), np.array(direction)))
            # If it goes off the edge, ignore the action
            if new_xy[0] < 0 or new_xy[0] >= len(self.map[0]):
                continue
            elif new_xy[1] < 0 or new_xy[1] >= len(self.map):
                continue

            actions.append(direction)

        return actions

    def is_obstacle(self, xy: Tuple[int, int]) -> bool:
        return self.map[xy[1]][xy[0]] == 1

    def get_neighbors(
        self, xy: Tuple[int, int]
    ) -> Tuple[int, int, int, int, int, int, int, int]:
        """
        get_neighbors takes a given xy coordinate, then returns the
        neighbors in order from top around clockwise:
        812
        7#3
        654
        """
        neighbors = []
        for direction in self.get_actions(xy):
            new_xy = tuple(np.add(np.array(xy), np.array(direction)))
            neighbors.append(new_xy)
        return neighbors

    def generate_move_probabilities(
        self, xy: Tuple[int, int]
    ) -> Dict[Direction, float]:
        """
        generate_step_policy_stochastic takes a given xy coordinate and, based
            on possible directions to go in, and the current value map,
            specifies the probability that each possible neighbor is chosen.
            Probabilities add to 1 and are appropriately scaled based on our
            rules.

            The rules specified for this scenario are:
            1. Choose the highest neighboring possible possible (evenly split
                if multiple neighbors tie)
            2. 20% chance of choosing an adjacent (45 degree rotation) neighbor
                instead.
        """
        neighbors = self.get_neighbors(xy)
        rewards = []

        # First we must calculate the resulting awards based on what we know for
        # each neighbor
        for neighbor in neighbors:
            # First, is the neighbor the goal? If so, it's worth REWARD points
            if neighbor == self.goal:
                rewards.append(GOAL)
            elif self.is_obstacle(neighbor):
                rewards.append(OBSTACLE_PENALTY)
            else:
                rewards.append(self.values[neighbor] - MOVEMENT_PENALTY)

        # Now that we have the rewards for each possible neighbor, we will figure
        # out the probability of choosing each.
        # To do this, first we find our highest score. This is possibly *all* of
        # them if all optons are equal.
        highest = [1 if x == max(rewards) else 0 for x in rewards]
        # Probabilities is a map, where the default value is a set [0] the length
        # of our neighbor count
        probabilities = defaultdict(lambda: [0] * len(neighbors))
        for index, high in enumerate(highest):
            if high:
                xy = neighbors[index]
                # Find what a 45 degree turn would be. Since we list the array
                # clockwise, this would be +/- 1 with index bounds wrapping
                left = index - 1
                if left < 0:
                    left = len(neighbors) - 1
                right = index + 1
                if right >= len(neighbors):
                    right = 0
                # Create a list of probabilities for each possible neighbor
                probabilities[xy] = [0.0] * len(neighbors)
                probabilities[xy][index] = 0.8
                # Our turn percentage is 20% total, or 10% chance each.
                probabilities[xy][left] = 0.1
                probabilities[xy][right] = 0.1

        return probabilities

    def generate_value_map(self, min_delta=1e-3):
        """
        generate_value_map will iteratively generate the calculated
            value of each non obstacle square given the starting
            conditions/state. It finishes when a termination minimum
            delta is hit
        """
        # values is our value map. It is a defaultdict of each xy
        # coordinate and its corresponding calculated value. We use
        # the default dict to set our default value to 0 appropriately.
        self.values = defaultdict(lambda: 0.0)
        # self.values[self.goal] = GOAL

        max_delta = min_delta + 1

        while max_delta > min_delta:
            # Reset max delta
            max_delta = 0.0

            for y, row in enumerate(self.map):
                for x, cell in enumerate(row):
                    xy = (x, y)
                    current_value = self.values[xy]

                    # Skip calculations on obstacles
                    if self.is_obstacle(xy):
                        # self.values[xy] = OBSTACLE_PENALTY
                        continue

                    # Calculate our value given our current position.
                    # To do this, we need our current value (somewhat
                    # known), the values of each possible neighbor,
                    # and
                    neighbors = self.get_neighbors(xy)

                    # Calculate the given probability of moving to each
                    # neighboring state based on our rules.
                    probabilities = self.generate_move_probabilities(xy)

                    # For each neighbor, multiply the value of all neighbors
                    # against the calculated probabilities. This allows us to
                    # do complicated multi-cell calcs like our 20% turn, or
                    # other ones like "10% of going in reverse", etc
                    neighbor_values = []
                    for neighbor in neighbors:
                        value = 0.0
                        for index, cell_xy in enumerate(neighbors):
                            value += (
                                current_value
                                + probabilities[neighbor][index] * self.values[cell_xy]
                            )
                        neighbor_values.append(value)

                    # Our current reward value is the highest possible reward
                    # choice of the calcultaed policy
                    self.values[xy] = max(neighbor_values)

                    # Assign the delta so that we can possibly escape later if
                    # we are barely updating anymore
                    delta = abs(self.values[xy] - max(neighbor_values))
                    max_delta = max(max_delta, delta)

            self.print_value_map()
            raise "hey now"


x = Map()
x.generate_value_map()
