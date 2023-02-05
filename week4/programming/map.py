import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

OPEN = 0
OBSTACLE = 1
ROBOT = 2

Direction = Tuple[int, int]
U: Direction = (0, -1)
UR: Direction = (1, -1)
R: Direction = (1, 0)
DR: Direction = (1, 1)
D: Direction = (0, 1)
DL: Direction = (-1, 1)
L: Direction = (-1, 0)
UL: Direction = (-1, -1)
directions = [U, UR, R, DR, D, DL, L, UL]

Coordinate = Tuple[int, int]

GOAL = 100.0
OBSTACLE_PENALTY = -50.0
MOVEMENT_PENALTY = -1.0


class Map:
    def __init__(self, goal: Tuple[int, int] = (10, 7), robot=(40, 7)):
        self.goal = goal
        self.robot = robot
        self.__init_map__()

        self.value_map = defaultdict(lambda: 0.0)
        self.policy_map = defaultdict(lambda: [0.125] * 8)

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
        return (len(self.map[0]), len(self.map))

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
                # if self.goal == xy:
                #     print(str(int(GOAL)), end=" ")
                else:
                    print(str(int(self.value_map[xy])).zfill(3), end=" ")
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

    # def get_actions(self, xy: Tuple[int, int]) -> List[Direction]:
    #     """
    #     get_actions takes a given xy coordinate, then returns the
    #     possible open actions available in the order relative to
    #     xy clockwise:
    #     812
    #     7#3
    #     654
    #     """
    #     actions = []
    #     for direction in directions:
    #         new_xy = tuple(np.add(np.array(xy), np.array(direction)))
    #         # If it goes off the edge, ignore the action
    #         if new_xy[0] < 0 or new_xy[0] >= len(self.map[0]):
    #             continue
    #         elif new_xy[1] < 0 or new_xy[1] >= len(self.map):
    #             continue

    #         actions.append(direction)

    #     return actions

    def is_obstacle(self, xy: Tuple[int, int]) -> bool:
        if self.is_out_of_bounds(xy):
            return False
        else:
            return self.map[xy[1]][xy[0]] == 1

    def is_out_of_bounds(self, xy: Tuple[int, int]) -> bool:
        return (
            xy[0] < 0
            or xy[0] >= len(self.map[0])
            or xy[1] < 0
            or xy[1] >= len(self.map)
        )

    def get_neighbors(self, xy: Tuple[int, int]) -> List[Coordinate]:
        """
        get_neighbors takes a given xy coordinate, then returns the
        neighbors in order from top around clockwise:
        812
        7#3
        654
        """
        neighbors = []
        for direction in directions:  # self.get_actions(xy):
            new_xy = tuple(np.add(np.array(xy), np.array(direction)))
            neighbors.append(new_xy)
        return neighbors

    def get_rewards(self, xy: Tuple[int, int]) -> List[float]:
        """
        generate_rewards takes a given xy coordinate and returns the reward
        of each adjacent move. Note that this is not the value, but rather
        the raw reward of moving to that cell.

        :returns List[float] - a list of rewards in our typical clockwise
            order
        """
        neighbors = self.get_neighbors(xy)
        rewards = []

        # First we must calculate the resulting awards based on what we know for
        # each neighbor
        for neighbor in neighbors:
            # First, is the neighbor the goal? If so, it's worth REWARD points
            if neighbor == self.goal:
                rewards.append(GOAL + MOVEMENT_PENALTY)
            elif self.is_obstacle(neighbor):
                rewards.append(OBSTACLE_PENALTY)
            else:
                rewards.append(MOVEMENT_PENALTY)

        return rewards

    def generate_move_probabilities_old(
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
        # Generate our neighbors
        neighbors = self.get_neighbors(xy)

        # Get our list of rewards per neighbor
        rewards = self.get_rewards(xy)
        values = [self.values[xy] for neighbor in neighbors]
        rewards = np.add(rewards, values)

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

    def generate_move_probabilities(self, xy: Tuple[int, int]) -> List[float]:
        """
        TODO - write me
        """
        # Generate our neighbors
        neighbors = self.get_neighbors(xy)

        # Get our list of rewards per neighbor
        values = [0.0] * len(neighbors)
        for index, neighbor in enumerate(neighbors):
            if self.is_out_of_bounds(neighbor):
                values[index] = -sys.maxsize
            elif self.is_obstacle(neighbor):
                values[index] = OBSTACLE_PENALTY
            elif neighbor == self.goal:
                values[index] = GOAL
            else:
                # values[index] = MOVEMENT_PENALTY + self.value_map[neighbor]
                values[index] = self.value_map[neighbor]

        # Now that we have the rewards for each possible neighbor, we will figure
        # out the probability of choosing each.
        # To do this, first we find our highest score. This is possibly *all* of
        # them if all optons are equal.
        highest = [1 if value == max(values) else 0 for value in values]
        # Probabilities is a map, where the default value is a set [0] the length
        # of our neighbor count
        probabilities = [1 / highest.count(1) * x for x in highest]

        return probabilities

    def policy_evaluation(self, gamma=0.95, theta=1e-3):
        # First we calculate the value map for the given policy
        delta = theta + 1
        while delta > theta:
            delta = 0

            for y, row in enumerate(self.map):
                for x, cell in enumerate(row):
                    # We ignore calculating for obstacle squares since
                    # we can never be there
                    if cell == OBSTACLE:
                        continue

                    xy = (x, y)

                    if xy == self.goal:
                        continue

                    # Save old value
                    old_value = self.value_map[xy]

                    # Get our neighbors, the probability of choosing them
                    # via our policy, and the rewards from each possible
                    # neighbor
                    neighbors = self.get_neighbors(xy)
                    probability = self.policy_map[xy]
                    rewards = self.get_rewards(xy)

                    # Calculate value by iterating over each and calculating
                    # the new values
                    value = 0.0
                    for index, neighbor in enumerate(neighbors):
                        # Ignore out of bounds neighbors - they don't exist
                        # So therefore we do not update their value at all
                        if self.is_out_of_bounds(neighbor):
                            continue
                        # You can't move to an obstacle state, so it has no value
                        # but if you do choose to try you take the obstacle
                        # penalty - therefore we do add its probability *
                        # the penalty as part of the reward for this state given
                        # our policy atm
                        if self.is_obstacle(neighbor):
                            value += probability[index] * rewards[index]
                        else:
                            value += probability[index] * (
                                rewards[index] + gamma * self.value_map[neighbor]
                            )

                    # Assign the new value
                    self.value_map[xy] = value
                    # See if our delta is the largest delta for this iteration
                    delta = max(delta, abs(old_value - value))

    def policy_iteration(self, gamma=0.95, theta=1e-3):
        # Now we update the polciy versus the value map we've calculated
        policy_stable = False
        iteration = 0
        changed = 0

        while not policy_stable:
            iteration += 1
            print(
                f"Iteration: {iteration} - Changed Policies = {changed}",
                # end="\r",
                # flush=True,
            )

            changed = 0
            policy_stable = True

            # Update our value map with our current policy
            self.policy_evaluation(gamma, theta)

            # Now we do policy improvement
            for y, row in enumerate(self.map):
                for x, cell in enumerate(row):
                    # Skip obstacles
                    if cell == OBSTACLE:
                        continue

                    xy = (x, y)

                    old_policy = self.policy_map[xy]
                    self.policy_map[xy] = self.generate_move_probabilities(xy)
                    policy_stable = policy_stable and old_policy == self.policy_map[xy]
                    if old_policy != self.policy_map[xy]:
                        changed += 1

            self.print_value_map()


x = Map()
x.policy_iteration()
x.print_value_map()
