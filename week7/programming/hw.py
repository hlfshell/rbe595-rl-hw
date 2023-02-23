from collections import defaultdict
from random import choice, random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

OPEN = 0
CLIFF = 1
ROBOT = 2

Direction = Tuple[int, int]
U: Direction = (0, -1)
R: Direction = (1, 0)
D: Direction = (0, 1)
L: Direction = (-1, 0)
directions: List[Direction] = [U, R, D, L]

Coordinate = Tuple[int, int]


class Map:
    def __init__(self, goal: Coordinate = (11, 3), robot: Coordinate = (0, 3)):
        self.goal = goal
        self.robot = robot
        self.__init_map__()

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

    def get(self, xy: Coordinate) -> int:
        return self.map[xy[1]][xy[0]]

    def shape(self) -> Coordinate:
        return (len(self.map[0]), len(self.map))

    def set_robot(self, xy):
        self.robot = xy

    def move_robot(self, direction: Direction) -> Tuple[int, Coordinate, bool]:
        xy = tuple(np.add(np.array(self.robot), np.array(direction)))
        if self.is_out_of_bounds(xy):
            raise Exception("Illegal move - out of bounds")
        else:
            self.set_robot(xy)

            if self.robot == self.goal:
                return (-1, self.robot, True)
            elif self.is_cliff(self.robot):
                return (-100, self.robot, True)
            else:
                return (-1, self.robot, False)

    def is_cliff(self, xy: Coordinate) -> bool:
        if self.is_out_of_bounds(xy):
            return False
        else:
            return self.map[xy[1]][xy[0]] == 1

    def is_out_of_bounds(self, xy: Coordinate) -> bool:
        return (
            xy[0] < 0
            or xy[0] >= len(self.map[0])
            or xy[1] < 0
            or xy[1] >= len(self.map)
        )

    def get_neighbors(self, xy: Coordinate) -> Tuple[List[Coordinate], List[Direction]]:
        """
        get_neighbors takes a given xy coordinate, then returns the
        neighbors in order from top around clockwise:
         1
        4#2
         3
        It also returns the list of directions allowed for those
        neighbors
        """
        allowed_neighbors = []
        allowed_directions = []
        for direction in directions:
            new_xy = tuple(np.add(np.array(xy), np.array(direction)))
            if self.is_out_of_bounds(new_xy):
                continue
            allowed_neighbors.append(new_xy)
            allowed_directions.append(direction)
        return allowed_neighbors, allowed_directions

    def print(self):
        for row in self.map:
            for col in row:
                print(col, end="")
            print()

    def draw(self, pixels_per=25):
        w, h = self.shape()
        img = Image.new("RGB", (w * pixels_per, h * pixels_per))
        draw = ImageDraw.Draw(img)

        for r, row in enumerate(self.map):
            for c, cell in enumerate(row):
                if (c, r) == self.goal:
                    color = "red"
                # elif (c, r) == self.robot:
                #     color = "green"
                elif cell == OPEN:
                    color = "white"
                elif cell == CLIFF:
                    color = "black"

                upper_left = (c * pixels_per, r * pixels_per)
                bottom_right = (upper_left[0] + pixels_per, upper_left[1] + pixels_per)
                draw.rectangle([upper_left, bottom_right], color)

        return img


class Agent:
    def __init__(self, alpha: float = 0.10, gamma: float = 0.95, epsilon=0.05):
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon

        self.reset_map()

        self.q: Dict[Tuple(Coordinate, Direction), float] = defaultdict(lambda: 0.0)

    def reset_map(self):
        self.map = Map()

    def policy(self, state: Coordinate) -> Direction:
        """
        Policy is an argmax epsilon-greedy function. Given a state, compare all available
        directions and expected Q values for them, then select the action with the highest
        Q-score. This is an epsilon-greedy approach, so we do this at a probability of
        1-epsilon. If it doesn't, then we just choose a random action instead.

        If multiple neighbors have equivalent value, then we choose randomly amongst them.
        """
        # We need to determine first what directions/neighbors are available to avoid
        # allowing going off the edge of our map
        _, actions = self.map.get_neighbors(state)

        # Then we determine if we're going to be choosing a random direction or
        # using our greedy approach of max Q value
        if random() > 1 - self.epsilon:
            return choice(actions)
        else:
            q_scores = [self.q[(state, direction)] for direction in actions]
            q_scores.sort(reverse=True)
            max_q_score = q_scores[0]
            # Limit directions to all values equivalent to the max Q score
            actions = [
                action for action in actions if self.q[(state, action)] == max_q_score
            ]
            return choice(actions)

    def episode(self):
        """
        Run through one episode and update according to the chosen algorithm
        """
        states: List[Coordinate] = []
        actions: List[Direction] = []
        state_action_pairs: List[Tuple[Coordinate, Direction]] = []
        rewards: List[int] = []

        terminal = False
        while not terminal:
            state = self.map.robot
            states.append(state)
            action = self.policy(state)

            reward, _, terminal = self.map.move_robot(action)

            actions.append(action)
            rewards.append(reward)
            state_action_pair = (state, action)
            state_action_pairs.append(state_action_pairs)

            next_state = self.map.robot
            next_action = self.policy(next_state)
            next_state_action_pair = (next_state, next_action)

            self.q[state_action_pair] = self.q[state_action_pair] + self.alpha * (
                reward
                + self.gamma * self.q[next_state_action_pair]
                - self.q[state_action_pair]
            )

    def run(self, episodes: int):
        """
        Perform a run of several episodes, allowing the agent to learn
        """
        for i in range(episodes):
            self.reset_map()
            self.episode()

    def draw_policy_map(self, pixels_per=25):
        img = self.map.draw(pixels_per=pixels_per)

        arrows = defaultdict(lambda: Image())
        arrow = Image.open("./arrow.png")
        for index, action in enumerate(directions):
            arrows[action] = arrow.rotate(-90 * index)

        for y, row in enumerate(self.map.map):
            for x, cell in enumerate(row):
                xy = (x, y)
                if xy == self.map.goal:
                    continue
                elif cell == CLIFF:
                    continue

                # Define the direction we go from here by finding the max
                # of the policy currently assigned
                _, actions = self.map.get_neighbors(xy)
                actions.sort(
                    key=lambda direction: self.q[(xy, direction)], reverse=True
                )
                action = actions[0]

                arrow = arrows[action]
                upper_left = (x * pixels_per, y * pixels_per)
                img.paste(arrow, upper_left)

        return img


if __name__ == "__main__":
    agent = Agent()
    agent.run(10)
    # map = agent.draw_policy_map()
    # map.save("./sarasa_10.png")
