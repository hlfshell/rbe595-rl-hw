import colorsys
import sys
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

DETERMINISTIC = "deterministic"
STOCHASTIC = "stochastic"
METHODS = [DETERMINISTIC, STOCHASTIC]


class Map:
    def __init__(self, goal: Tuple[int, int] = (10, 7), robot=(40, 7)):
        self.goal = goal
        self.robot = robot
        self.__init_map__()

        self.value_map = defaultdict(lambda: 0.0)
        self.policy_map = defaultdict(lambda: [0.125] * 8)

    def __init_map__(self):
        self.map: List[List[int]] = []

        with open("./map.dat", "r") as file:
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

    def draw(self, pixels_per=25):
        w, h = self.shape()
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

    def draw_value_map(self, pixels_per=25):
        img = self.draw(pixels_per=pixels_per)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        for y, row in enumerate(self.map):
            for x, cell in enumerate(row):
                xy = (x, y)
                if xy == self.goal:
                    continue
                elif cell == OBSTACLE:
                    continue
                upper_left = (x * pixels_per, y * pixels_per)
                bottom_right = (upper_left[0] + pixels_per, upper_left[1] + pixels_per)
                draw.rectangle(
                    [upper_left, bottom_right],
                    fill=self._value_to_color(self.value_map[xy]),
                )
                text_x = int(upper_left[0] + (pixels_per * 0.3))
                text_y = int(bottom_right[1] - (pixels_per * 0.66))
                draw.text((text_x, text_y), str(round(self.value_map[xy])), font=font)

        return img

    def _value_to_color(self, value: int):
        """
        _value_to_color takes a given value and converts it to a color based on the
            current min and max values in self.value_map
        """
        in_max = GOAL
        in_min = min(self.value_map.values())
        # (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
        mapped_value = (value - in_min) * (120 - 0) / (in_max - in_min) + 0
        conversion = colorsys.hls_to_rgb(mapped_value / 360, 0.5, 1.0)
        return (
            int(conversion[0] * 255),
            int(conversion[1] * 255),
            int(conversion[2] * 255),
        )

    def set_robot(self, xy):
        self.robot = xy

    def move_robot(self, direction: Direction):
        xy = tuple(np.add(np.array(self.robot), np.array(direction)))
        if self.get(xy) == 1:
            raise Exception("Illegal move")
        else:
            self.set_robot(xy)

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

    def generate_move_probabilities(
        self, xy: Tuple[int, int], method: str
    ) -> List[float]:
        """
        generate_move_probabilities will accept a given xy coordinate and, with the given
            reward structure and calculated value map, deccide what the probability of
            moving to any given neighbor is. The function switches the behaviour from
            deterministic (simply the max score, split evnly if tied) and stochastic
            (the same as before, but with a 20% chance at veering 45 degrees to the left
            or right).

            Note that out of bounds neighbors are given a maximum negative penalty, as we
            need to ensure that we eliminate it as a possibility.

            The two methods are either DETERMINISTIC or STOCHASTIC

            :returns List[float] - a list of floats in clockwise directional order of the
                probability the agent would want to move to the given neighbor
        """
        if method not in METHODS:
            raise Exception(f"Invalid method chosen - must be one of: {METHODS}")

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
        if method == DETERMINISTIC:
            # To do this, first we find our highest score. This is possibly *all* of
            # them if all optons are equal.
            highest = [1 if value == max(values) else 0 for value in values]
            # Probabilities is a map, where the default value is a set [0] the length
            # of our neighbor count
            probabilities = [1 / highest.count(1) * x for x in highest]
        else:
            # Here we want to find the max (and all equivalent), and then
            # include a 20% chance at taking a 45 degree turn. Take note that
            # this means that there can be overlap between possible outcomes.
            highest = [1 if value == max(values) else 0 for value in values]
            # Next we define all the turn-off indexes
            turns = [0] * len(highest)
            for index, high in enumerate(highest):
                if not high:
                    continue
                left = index - 1
                if left < 0:
                    left = len(neighbors) - 1
                right = index + 1
                if right >= len(neighbors):
                    right = 0
                turns[left] += 1
                turns[right] += 1
            # Note that there can be multiple turns now.
            totals = np.add([x * 0.8 for x in highest], [x * 0.1 for x in turns])
            probabilities = [x / sum(totals) for x in totals]

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

    def policy_iteration(self, method: str, gamma=0.95, theta=1e-3):
        # Now we update the polciy versus the value map we've calculated
        policy_stable = False
        iteration = 0
        changed = 0

        while not policy_stable:
            iteration += 1
            print(
                f"Iteration: {iteration} - Changed Policies = {changed}",
                end="\r",
                flush=True,
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
                    self.policy_map[xy] = self.generate_move_probabilities(xy, method)
                    policy_stable = policy_stable and old_policy == self.policy_map[xy]
                    if old_policy != self.policy_map[xy]:
                        changed += 1


if __name__ == "__main__":
    map = Map()

    map.draw().save("./imgs/map.png")

    print("Policy iteration for determinstic policy:")
    map.policy_iteration(DETERMINISTIC)
    map.draw_value_map().save("./imgs/policy_iteration_value_map_deterministic.png")

    print("Policy iteration for stochastic policy:")
    map.policy_iteration(STOCHASTIC)
    map.draw_value_map().save("./imgs/policy_iteration_value_map_stochastic.png")
