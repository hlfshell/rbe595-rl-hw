from collections import defaultdict
from random import choice, random
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

OPEN = 0
OBSTACLE = 1
ROBOT = 2

Direction = Tuple[int, int]
U: Direction = (0, -1)
R: Direction = (1, 0)
D: Direction = (0, 1)
L: Direction = (-1, 0)
directions: List[Direction] = [U, R, D, L]

Coordinate = Tuple[int, int]


class Map:
    def __init__(self, goal: Coordinate = (8, 0), robot: Coordinate = (0, 2)):
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
        """
        move_robot moves the current robot in the direction specified. Returns
        the reward, the new current state, and if it was a terminal state
        """
        xy = tuple(np.add(np.array(self.robot), np.array(direction)))
        if self.is_out_of_bounds(xy):
            raise Exception("Illegal move - out of bounds")
        else:
            if xy == self.goal:
                self.set_robot(xy)
                return (1, self.robot, True)
            elif self.is_obstacle(xy):
                return (0, self.robot, False)
            else:
                self.set_robot(xy)
                return (0, self.robot, False)

    def is_obstacle(self, xy: Coordinate) -> bool:
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
                elif cell == OBSTACLE:
                    color = "black"

                upper_left = (c * pixels_per, r * pixels_per)
                bottom_right = (upper_left[0] + pixels_per, upper_left[1] + pixels_per)
                draw.rectangle([upper_left, bottom_right], color)

        return img


class DynaQ:
    def __init__(
        self,
        planning_steps: int,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 0.05,
    ):
        self.planning_steps = planning_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.reset_map()

        self.q: Dict[Tuple[Coordinate, Direction], float] = defaultdict(lambda: 0.0)
        self.model: Dict[
            Coordinate, Dict[Direction, Tuple[Coordinate, float]]
        ] = defaultdict(lambda: {})

    def reset_map(self):
        self.map = Map()

    def qmax_action(self, state: Coordinate) -> Direction:
        """
        qmax_action will return the highest scored action from a given state, with ties
        being randomly chosen from amongst them. There is no randomness beyond this.
        """
        _, actions = self.map.get_neighbors(state)
        q_scores = [self.q[(state, direction)] for direction in actions]
        q_scores.sort(reverse=True)
        max_q_score = q_scores[0]
        # Limit directions to all values equivalent to the max Q score
        actions = [
            action for action in actions if self.q[(state, action)] == max_q_score
        ]
        return choice(actions)

    def policy(self, state: Coordinate) -> Direction:
        """
        policy is an argmax epsilon-greedy function. Given a state, compare all available
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
            return self.qmax_action(state)

    def episode(self) -> float:
        terminal = False

        while not terminal:
            state = self.map.robot
            action = self.policy(state)
            reward, new_state, terminal = self.map.move_robot(action)
            next_best_action = self.qmax_action(new_state)
            self.q[(state, action)] = self.q[(state, action)] + self.alpha * (
                reward
                + self.gamma * self.q[(new_state, next_best_action)]
                - self.q[(state, action)]
            )
            self.model[state][action] = (new_state, reward)

            self.planning()

    def planning(self):
        # Planning steps; propagate the information we just learned
        # across as many planning steps we can, up to the agent's
        # specified maximum. To be called via the episode function
        n = min(len(self.q.keys()), self.planning_steps)
        n = self.planning_steps
        for _ in range(n):
            state = choice(list(self.model.keys()))
            action = choice(list(self.model[state].keys()))
            new_state, reward = self.model[state][action]
            next_best_action = self.qmax_action(new_state)
            self.q[(state, action)] = self.q[(state, action)] + self.alpha * (
                reward
                + self.alpha * self.q[new_state, next_best_action]
                - self.q[(state, action)]
            )

    def run(self, episodes: int) -> List[float]:
        """
        Perform a run of several episodes, allowing the agent to learn
        Returns a list of all rewards collected during the episodes.
        """
        rewards: List[float] = []
        for i in range(episodes):
            # print(i + 1, end="\r", flush=True)
            self.reset_map()
            reward = self.episode()
            rewards.append(reward)

        # print("")

        return rewards

    def steps_to_solve(self) -> float:
        """
        Given our current learned policy, what are the # of steps it
        would take our agent to solve the problem?
        """
        self.reset_map()
        steps = 0
        terminal = False
        while not terminal:
            steps += 1
            state = self.map.robot
            action = self.qmax_action(state)
            _, _, terminal = self.map.move_robot(action)

        return steps

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
                elif cell == OBSTACLE:
                    continue

                action = self.qmax_action(xy)

                arrow = arrows[action]
                upper_left = (x * pixels_per, y * pixels_per)
                img.paste(arrow, upper_left)

        return img


if __name__ == "__main__":
    steps: Dict[int, List[float]] = {}
    for planning_steps in [0, 5, 50]:
        print(f"Calculating for agent with {planning_steps} planning steps...")
        steps[planning_steps] = []
        for episodes in range(50):
            agent = DynaQ(planning_steps=planning_steps)
            agent.run(episodes)
            steps[planning_steps].append(agent.steps_to_solve())
        agent.draw_policy_map().save(f"./imgs/{planning_steps}_policy_map.png")

    figure, axes = plt.subplots()
    pos = axes.get_position()
    episodes_axis = [i + 1 for i in range(50)]
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.plot(episodes_axis, steps[0], label=0)
    plt.plot(episodes_axis, steps[5], label=5)
    plt.plot(episodes_axis, steps[50], label=50)

    axes.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    axes.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))

    figure.savefig("imgs/steps_episodes_full.png")

    # Create a chart excluding the first episode
    figure, axes = plt.subplots()
    pos = axes.get_position()
    episodes_axis = [i + 1 for i in range(50)]
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.plot(episodes_axis[2:], steps[0][2:], label=0)
    plt.plot(episodes_axis[2:], steps[5][2:], label=5)
    plt.plot(episodes_axis[2:], steps[50][2:], label=50)

    axes.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    axes.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))

    figure.savefig("imgs/steps_episodes_sans_episode_1.png")
