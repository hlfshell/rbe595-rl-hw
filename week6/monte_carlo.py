from collections import defaultdict
from random import randint
from typing import Dict, List, Tuple


class Map:
    """
    A map object represents the world w/ our simplistic
    setup. The map is determinstic, so we'll simply code
    the behaviours into here.
    """

    def __init__(self):
        # Our robot will always start inposition 3
        self.robot: int = 0
        self.terminal: bool = False
        self.reset()

    def reset(self):
        """
        reset resets the simulation and places the robot
        in a random spot
        """
        self.robot = randint(1, 4)

    def take_action(self, step: int) -> Tuple[int, int, bool]:
        """
        Given an action U, return the reward
        of the robot's action, its new position,
        and if it's a terminal state
        """
        # Deal first with our terminal states - we can't move from it
        if self.robot == 0 or self.robot == 5:
            return (0, self.robot, True)

        reward = 0
        terminal = False

        if self.robot == 1 and step == -1:
            reward = 1
            terminal = True
        elif self.robot == 4 and step == 1:
            reward = 5
            terminal = True

        self.robot = self.robot + step

        return (reward, self.robot, terminal)


class MonteCarloExploringStarts:
    def __init__(self, map: Map):
        self.map = map
        self.returns: Dict[Tuple, List[float]] = defaultdict(lambda: [])
        self.q: Dict[Tuple, float] = defaultdict(lambda: 0.0)

    def policy(self, state: int) -> int:
        """
        Given a state, return the action U w/ the policy of
        argmax(Q(St, a)) between the two possible acions of
        -1 and 1
        """
        return 1 if self.q[state, 1] > self.q[state, -1] else -1

    def episode(self, gamma: float = 0.95):
        """
        Run through one episode
        """
        self.map.reset()
        terminal = self.map.terminal
        states: List[int] = []
        actions: List[int] = []
        state_action_pairs: List[Tuple[int, int]] = []
        rewards: List[int] = []

        while not terminal:
            state = self.map.robot
            states.append(state)

            action = self.policy(self.map.robot)
            reward, _, terminal = self.map.take_action(action)

            actions.append(action)
            rewards.append(reward)
            state_action_pairs.append((state, action))

        # By this point we have reached a terminal state. Let us
        # begin calculating our G and adjusting our returns + q
        # We iterate over each state from terminal in reverse,
        # skipping our terminal state. Note that if we started on
        # a terminal this is empty, so we just skip this
        G = 0
        for index in range(len(states) - 1 - 1, 0 - 1, -1):
            G = gamma * G + rewards[index + 1]

            # If we do not see this state action pair earlier than
            # at this point, we can safely continue
            state_action_pair = (states[index], actions[index])
            if index == state_action_pairs.index(state_action_pair):
                self.returns[state_action_pair].append(G)
                self.q[state_action_pair] = sum(self.returns[state_action_pair]) / len(
                    self.returns[state_action_pair]
                )

    def run(self, episodes: int):
        """
        Perform a run of several episodes, allowing the agent to learn
        """
        for i in range(episodes):
            self.episode()


if __name__ == "__main__":
    map = Map()
    agent = MonteCarloExploringStarts(map)
    agent.run(100)
    print(len(agent.returns))
    print(agent.q)
