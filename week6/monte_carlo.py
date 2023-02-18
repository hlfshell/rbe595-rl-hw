from collections import defaultdict
from random import randint, choice, random
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt

EXPLORING_START = 0
FIRST_VISIT = 1


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


class MonteCarlo:
    def __init__(self, map: Map, agent_type: int, epsilon=0.05):
        self.map = map
        self.agent_type = agent_type
        self.epsilon = epsilon

        if self.agent_type not in [EXPLORING_START, FIRST_VISIT]:
            raise Exception("Invalid agent type specified")

        self.returns: Dict[Tuple, List[float]] = defaultdict(lambda: [])
        self.q: Dict[Tuple, float] = defaultdict(lambda: 0.0)

    def policy(self, state: int) -> int:
        if self.agent_type == EXPLORING_START:
            return self.exploring_start_policy(state)
        elif self.agent_type == FIRST_VISIT:
            return self.first_visit_policy(state)
        else:
            raise Exception("Invalid agent type specified")

    def exploring_start_policy(self, state: int) -> int:
        """
        Given a state, return the action U w/ the policy of
            argmax(Q(St, a)) between the two possible acions of
            -1 and 1. On a tie, pick one at random.

        """
        right = (state, 1)
        left = (state, -1)

        if self.q[left] > self.q[right]:
            return -1
        elif self.q[left] < self.q[right]:
            return 1
        else:
            return choice([-1, 1])

    def first_visit_policy(self, state: int) -> int:
        """
        Given a state, return the action U w/ the policy of
            argmax(Q(St, a)) between the two possible acions of
            -1 and 1. On a tie, pick one at random.

            Note that this agent has a given epsilon which is
                taken into account - there is a chance the
                policy will decide to explore rather than
                take the max Q value.

        :returns int - the direction to go
        """
        right = (state, 1)
        left = (state, -1)

        action = 0
        if self.q[left] > self.q[right]:
            action = -1
        elif self.q[left] < self.q[right]:
            action = 1
        else:
            action = choice([-1, 1])

        if 1 - self.epsilon >= random():
            return action
        else:
            return -1 * action

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

            action = self.policy(state)

            # Determine if the robot will listen to our action
            # or do someting else.
            if 0.8 >= random():
                performed_action = action

            # We are now split between other other two options
            # Remaining percentages given their ratios are 75%
            # and 25% as opposed to the original 15% and 5%
            elif 0.75 >= random():
                performed_action = 0
            else:
                performed_action = -1 * action

            reward, _, terminal = self.map.take_action(performed_action)

            actions.append(action)
            rewards.append(reward)
            state_action_pairs.append((state, action))

        # By this point we have reached a terminal state. Let us
        # begin calculating our G and adjusting our returns + q
        # We iterate over each state from terminal in reverse,
        # skipping our terminal state.
        G = 0

        for index in range(len(states) - 1, 0 - 1, -1):
            G = gamma * G + rewards[index]

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


def run_experiment(agent_type: int, episodes_max: int, runs_per_episode: int):
    map = Map()

    exploring_avg_qs: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(
        lambda: defaultdict(lambda: 0.0)
    )

    for episodes in range(1, episodes_max + 1):
        for _ in range(runs_per_episode):
            agent = MonteCarlo(map, agent_type)
            agent.run(episodes)
            for key in agent.q.keys():
                exploring_avg_qs[episodes][key] += agent.q[key] / runs_per_episode

    return exploring_avg_qs


def build_plots(results: Dict[int, Dict[Tuple[int, int], float]], filename: str):
    figure, axes = plt.subplots()
    plt.xlabel("Episodes Ran")
    plt.ylabel("Average Q Score")

    episodes_axis = [i + 1 for i in range(len(results.keys()))]
    for state_action in results[list(results.keys())[0]].keys():
        scores: List[float] = [
            results[episode][state_action] for episode in results.keys()
        ]
        plt.plot(episodes_axis, scores, label=str(state_action))

    pos = axes.get_position()
    axes.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    axes.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))

    figure.savefig(filename)


if __name__ == "__main__":
    results = run_experiment(EXPLORING_START, 500, runs_per_episode=100)
    build_plots(results, "./results_exploring_start.png")
    results = run_experiment(FIRST_VISIT, 500, runs_per_episode=100)
    build_plots(results, "./results_first_visit.png")
