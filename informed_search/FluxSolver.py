# Model
# Generate Grid

import random
import numpy as np


# Create World State
# Retrieve Slice
# Sort Slice
# Delete Slice
# Insert Slice

# Find all clusters


class WorldState:
    def __init__(self, size_row, size_col):
        # Start with the problem
        self.state = np.int_(np.array([
            [1, 1, 1, 2, 2, 2, 1, 1, 2],
            [1, 2, 2, 2, 2, 2, 1, 1, 2],
            [1, 1, 1, 2, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 1, 2, 1, 2],
            [2, 2, 2, 2, 1, 2, 2, 2, 1],
            [2, 1, 1, 2, 2, 1, 1, 2, 1],
        ]))

        # We want to set it so that the bottom left coordinate is 0, 0
        self.state = np.flipud(self.state)

        self.state = np.int_(np.array([
            [0, 1, 1, 2, 2, 2, 0, 1, 0],
            [0, 2, 0, 0, 0, 2, 0, 1, 2],
            [1, 1, 1, 2, 1, 0, 0, 2, 2],
            [1, 1, 2, 2, 2, 1, 0, 1, 2],
            [1, 2, 2, 2, 1, 2, 0, 2, 1],
            [2, 1, 1, 2, 2, 1, 0, 0, 1],
        ]))

    def bubble_up(self):
        for c in range(0, 9):
            col = self.state[:, c]
            nonz = col[col > 0]
            zeroCount = 6 - np.shape(nonz)[0]
            sorted = np.append(nonz, np.zeros(zeroCount))
            self.state = np.delete(self.state, c, 1)
            self.state = np.insert(self.state, c, sorted, 1)
        print(self.state)
        print()


world = WorldState(3, 2)
world.bubble_up()


def get_clusters(state):
    state_shape = np.shape(state)
    rows = state_shape[0]
    cols = state_shape[1]
    print(state_shape)

    indices = []
    for i in range(0, rows):
        for j in range(0, cols):
            indices.append(Coordinate(i, j))

    cluster_list = []
    indices = set(indices)
    while indices.__len__() > 0:
        node = list(indices).pop(0)
        cluster = get_cluster(state, node.row, node.col, state[node.row][node.col], set())
        cluster_list.append(cluster)
        indices.difference_update(cluster)

    return cluster_list


class Coordinate:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __hash__(self):
        return hash(f'{self.row},{self.col}')

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __str__(self):
        return f'({self.row}, {self.col})'

    def __repr__(self):
        return self.__str__()


def get_cluster(state, row, col, value, reached):
    if row < 0 or row >= 6:
        return {}
    if col < 0 or col >= 9:
        return {}

    if state[row][col] != value:
        return {}

    coords = Coordinate(row, col)
    if coords in reached:
        return {}

    reached.add(coords)

    up = get_cluster(state, row - 1, col, value, reached)
    reached.update(up)

    down = get_cluster(state, row + 1, col, value, reached)
    reached.update(down)

    left = get_cluster(state, row, col - 1, value, reached)
    reached.update(left)

    right = get_cluster(state, row, col + 1, value, reached)
    reached.update(right)

    return reached


l = get_clusters(world.state)
print(l)


# x = get_cluster(world.state, 0, 0, 1, set())
# print(x)


# print({Coordinate(0, 0), Coordinate(0, 0)})


class Node:
    def __init__(self, row, col, parent_state):
        self.row = row
        self.col = col
        self.parent_state = parent_state

    # def new_state(self):


class World:
    def __init__(self, size_row, size_col):
        self.size_row = size_row
        self.size_col = size_col
        self.state = [[0] * size_col for i in range(size_row)]
        self.points = []
        for col in range(size_col):
            for row in range(size_row):
                value = random.choices([0, 1], weights=[6, 4], k=1)[0]
                self.state[row][col] = value
                if value == 1:
                    self.points.append(Node(row, col, 2))

        # print(self.grid)

    def print(self):
        for row in self.state:
            print(f'{" | ".join(map(str, row))}')
        print("\n")


world = World(2, 2)


def eval_function(node):
    return 8


def greedy_best_first_search(world):
    frontier = []
    for p in world.points:
        frontier.append(p)

    frontier.sort(eval_function)
    return None

# World(2, 2).print()
