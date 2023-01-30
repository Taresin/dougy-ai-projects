# The game is this.
# You are given a grid of values.
# initial is the starting coordinate with a value of 0
# Find the shortest ascending path

import random
from enum import Enum


class Movement(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f'[{self.x}, {self.y}]'


class Node:
    def __init__(self, x, y):
        self.state = State(x, y)
        self.parent = None
        self.actions = []
        if x != 0:
            self.actions.append(Movement.LEFT)
        if x != 2:
            self.actions.append(Movement.RIGHT)
        if y != 0:
            self.actions.append(Movement.UP)
        if y != 2:
            self.actions.append(Movement.DOWN)
        self.cost = 0

    def __str__(self):
        return f'Node:\nState: {self.state}\nActions: {self.actions}]'


class Game:
    def __init__(self):
        self.goal = State(random.choice([0, 1, 2]), random.choice([0, 1, 2]))
        # self.goal = State(0, 1)
        cost_values = [1]
        self.grid = [
            random.choices(cost_values, k=3),
            random.choices(cost_values, k=3),
            random.choices(cost_values, k=3)
        ]
        self.grid[self.goal.x][self.goal.y] = 0
        self.initial = Node(0, 0)

    def print(self):
        for row in self.grid:
            print(f'{row[0]} | {row[1]} | {row[2]}')
        print("\n")

    def is_goal(self, node):
        return node.state.x == self.goal.x and node.state.y == self.goal.y


game = Game()
game.print()


def eval_function(node):
    x_diff = game.goal.x - node.state.x
    y_diff = game.goal.y - node.state.y
    return abs(x_diff) + abs(y_diff)


def expand(problem, node):
    s_dash = []
    for action in node.actions:
        match action:
            case Movement.LEFT:
                n = Node(node.state.x - 1, node.state.y)
                if node.state.x - 1 >= 0:
                    s_dash.append(n)
            case Movement.RIGHT:
                n = Node(node.state.x + 1, node.state.y)
                if node.state.x + 1 <= 2:
                    s_dash.append(n)
            case Movement.UP:
                n = Node(node.state.x, node.state.y - 1)
                if node.state.y - 1 >= 0:
                    s_dash.append(n)
            case Movement.DOWN:
                n = Node(node.state.x, node.state.y + 1)
                if node.state.y + 1 <= 2:
                    s_dash.append(n)
    return s_dash


def str(node):
    return node.__str__()


def best_first_search(problem, eval_function):
    node = problem.initial
    frontier = [node]
    reached = {node.state: node}
    while frontier.__len__() != 0:
        frontier.sort(key=eval_function, reverse=True)

        node = frontier.pop()
        print(node)
        problem.grid[node.state.x][node.state.y] = 'X'
        problem.print()

        if problem.is_goal(node):
            return node

        results = expand(problem, node)
        for child in results:
            key = child.state.__str__()
            if key not in reached.keys():
                reached[key] = child
                frontier.append(child)

    return None


node = best_first_search(game, eval_function)
print(f'\n\nThe goal node is found here\n{node}\n')
