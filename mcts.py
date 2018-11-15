import math
from tqdm import tqdm
from time import time

EXPLORE_PARAM = 2


class Node:
    def __init__(self, state, parent=None, action=None, is_terminal=False):
        self.state = state
        self.children = []
        self.parent = parent
        self.n_visits = 0
        self.reward = 0
        self.action = action
        self.is_terminal = is_terminal

    @property
    def value(self):
        """UCB1"""
        if self.n_visits == 0:
            return float('inf')
        return self.reward/self.n_visits + \
            EXPLORE_PARAM*math.sqrt(math.log(self.parent.n_visits)/self.n_visits)

    def best_child(self):
        return max(self.children, key=lambda n: n.value)


def mcts(root, expansion_policy, rollout_policy, iterations=2000, max_depth=200):
    """
    Monte Carlo Tree Search
    - `expansion_policy` should be a function that takes a node and returns a
    list of child nodes
    - `rollout_policy` should be a function that takes a node and returns a
    reward for that node
    """
    root.children = expansion_policy(root)

    # MCTS
    for _ in tqdm(range(iterations)):
        cur_node = root

        # Selection
        while True:
            if cur_node.n_visits > 0 and cur_node.children:
                cur_node = cur_node.best_child()
            else:
                break

        if cur_node.n_visits > 0:
            # If selection took us to a terminal node,
            # this seems to be the best path
            if cur_node.is_terminal:
                break

            # Expansion
            s = time()
            cur_node.children = expansion_policy(cur_node)
            print('Expansion took:', time() - s)
            cur_node = cur_node.best_child()

        # Rollout
        s = time()
        reward = rollout_policy(cur_node, max_depth=max_depth)
        print('Rollout took:', time() - s)

        # Update
        cur_node.reward += reward
        cur_node.n_visits += 1
        parent = cur_node.parent
        while parent is not None:
            parent.reward += reward
            parent.n_visits += 1
            parent = parent.parent

    # Return best path
    cur_node = root
    path = [cur_node]
    for _ in range(max_depth):
        cur_node = cur_node.best_child()
        path.append(cur_node)
        if cur_node.is_terminal:
            break

    # Max depth exceeded, no path found
    else:
        return None

    return path
