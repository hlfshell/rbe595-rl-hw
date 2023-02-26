from collections import defaultdict
from random import choice, random
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

OPEN = 0
CLIFF = 1
ROBOT = 2

SARSA = 0
QLEARNING = 1

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
    def __init__(
        self, type: int, alpha: float = 0.10, gamma: float = 0.95, epsilon=0.05
    ):
        self.type = type
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon

        self.reset_map()

        self.q: Dict[Tuple(Coordinate, Direction), float] = defaultdict(lambda: 0.0)

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
        """
        Run through one episode and update according to the chosen algorithm.
        Return the final reward of the episode.
        """
        states: List[Coordinate] = []
        actions: List[Direction] = []
        rewards: List[int] = []
        total_reward: float = 0.0

        terminal = False
        state = self.map.robot

        action = self.policy(state)
        while not terminal:
            state = self.map.robot
            states.append(state)

            if self.type == QLEARNING:
                action = self.policy(state)

            reward, _, terminal = self.map.move_robot(action)

            total_reward += reward
            actions.append(action)
            rewards.append(reward)
            state_action_pair = (state, action)

            next_state = self.map.robot
            if self.type == SARSA:
                next_action = self.policy(next_state)
            else:
                next_action = self.qmax_action(next_state)
            next_state_action_pair = (next_state, next_action)

            if self.type == SARSA:
                action = next_action

            self.q[state_action_pair] = self.q[state_action_pair] + self.alpha * (
                reward
                + self.gamma * self.q[next_state_action_pair]
                - self.q[state_action_pair]
            )

        return total_reward

    def run(self, episodes: int) -> List[float]:
        """
        Perform a run of several episodes, allowing the agent to learn
        Returns a list of all rewards collected during the episodes.
        """
        rewards: List[float] = []
        for i in range(episodes):
            print(i + 1, end="\r", flush=True)
            self.reset_map()
            reward = self.episode()
            rewards.append(reward)

        print("")

        return rewards

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

                action = self.qmax_action(xy)

                arrow = arrows[action]
                upper_left = (x * pixels_per, y * pixels_per)
                img.paste(arrow, upper_left)

        return img


if __name__ == "__main__":
    episodes = 10_000

    print("Running SARSA Agent")
    agent = Agent(SARSA, alpha=0.1, epsilon=0.05)
    sarsa_rewards = agent.run(episodes)
    map = agent.draw_policy_map()
    map.save("./imgs/sarsa.png")

    print("Running Q-Learning Agent")
    agent = Agent(QLEARNING, alpha=0.1, epsilon=0.05)
    qlearning_rewards = agent.run(episodes)
    map = agent.draw_policy_map()
    map.save("./imgs/qlearning.png")

    figure = plt.figure()
    episodes_axis = [i + 1 for i in range(episodes)]

    plt.plot(episodes_axis, sarsa_rewards)
    figure.savefig("imgs/sarsa_rewards.png")

    figure = plt.figure()

    plt.plot(episodes_axis, qlearning_rewards)
    figure.savefig("imgs/q_learning_rewards.png")

    print("Running Epsilon tests")
    # Run for lowering epsilons
    for epsilon in np.arange(0.05, 0.0 - 0.01, -0.01):
        epsilon = round(epsilon, 2)
        print(f"Epsilon = {epsilon}")

        print("Running SARSA Agent")
        agent = Agent(SARSA, alpha=0.1, epsilon=epsilon)
        agent.run(episodes)
        map = agent.draw_policy_map()
        map.save(f"./imgs/sarsa_epsilon_{epsilon}.png")

        print("Running Q-Learning Agent")
        agent = Agent(QLEARNING, alpha=0.1, epsilon=epsilon)
        agent.run(episodes)
        map = agent.draw_policy_map()
        map.save(f"./imgs/qlearning_epsilon_{epsilon}.png")
