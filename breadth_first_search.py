# BFS is one algorithm to find a solution to problems that are uninformed search.
# Best first search is good when you can tell how far you are from the goal.
# Some problems do not have this luxury so this algorithm gradually expands out the search

import random
from enum import Enum
import time
import os

GAME_SIZE = 6
DISPLAY_INTERVAL = 0.3


def cls():
    os.system('clear')


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
        if x > 0:
            self.actions.append(Movement.LEFT)
        if x < GAME_SIZE - 1:
            self.actions.append(Movement.RIGHT)
        if y > 0:
            self.actions.append(Movement.UP)
        if y < GAME_SIZE - 1:
            self.actions.append(Movement.DOWN)
        self.cost = 0

    def __str__(self):
        return f'Node:\nState: {self.state}\nActions: {self.actions}]'


class Game:
    def __init__(self):
        coord_range = range(0, GAME_SIZE, 1)
        self.goal = State(random.choice(coord_range), random.choice(coord_range))
        cost_values = [" "]
        self.grid = []
        for i in range(0, GAME_SIZE, 1):
            self.grid.append(random.choices(cost_values, k=GAME_SIZE))
        self.grid[self.goal.x][self.goal.y] = "0"
        self.initial = Node(random.choice(coord_range), random.choice(coord_range))
        self.grid[self.initial.state.x][self.initial.state.y] = "S"

    def print(self):
        for row in self.grid:
            print(f'{" | ".join(row)}')
            # print(f'{row[0]} | {row[1]} | {row[2]}')
        print("\n")

    def is_goal(self, node):
        return node.state.x == self.goal.x and node.state.y == self.goal.y

cls()
game = Game()
game.print()


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
                if node.state.x + 1 < GAME_SIZE:
                    s_dash.append(n)
            case Movement.UP:
                n = Node(node.state.x, node.state.y - 1)
                if node.state.y - 1 >= 0:
                    s_dash.append(n)
            case Movement.DOWN:
                n = Node(node.state.x, node.state.y + 1)
                if node.state.y + 1 < GAME_SIZE:
                    s_dash.append(n)
    return s_dash


def breadth_first_search(problem):
    node = problem.initial
    if problem.is_goal(node):
        return node
    frontier = [node]
    reached = {node.state.__str__()}
    while frontier.__len__() != 0:
        time.sleep(DISPLAY_INTERVAL)
        cls()
        node = frontier.pop(0)
        problem.grid[node.state.x][node.state.y] = "X"
        problem.print()
        print(node)
        if problem.is_goal(node):
            return child
        for child in expand(problem, node):
            s = child.state
            if s.__str__() not in reached:
                reached.add(s.__str__())
                frontier.append(child)
    return None


node = breadth_first_search(game)
print(f'\n\nThe goal node is found here\n{node}\n')
