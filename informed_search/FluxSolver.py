# Model
# Generate Grid

import random
import numpy as np
from copy import deepcopy

# Create World State
# Find all clusters

# Goal Formulation

initial_state = np.int_(np.array([
    [1, 1, 1, 2, 2, 2, 1, 1, 2],
    [1, 2, 2, 2, 2, 2, 1, 1, 2],
    [1, 1, 1, 2, 1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2, 1, 2, 1, 2],
    [2, 2, 2, 2, 1, 2, 2, 2, 1],
    [2, 1, 1, 2, 2, 1, 1, 2, 1],
]))
initial_state = np.flipud(initial_state)
initial_shape = np.shape(initial_state)
ROW_COUNT = initial_shape[0]
COL_COUNT = initial_shape[1]


class WorldState:
    def __init__(self, array=None, cluster_history=[]):
        # # Start with the problem
        # self.state = np.int_(np.array([
        #     [1, 1, 1, 2, 2, 2, 1, 1, 2],
        #     [1, 2, 2, 2, 2, 2, 1, 1, 2],
        #     [1, 1, 1, 2, 1, 1, 2, 2, 2],
        #     [1, 1, 2, 2, 2, 1, 2, 1, 2],
        #     [2, 2, 2, 2, 1, 2, 2, 2, 1],
        #     [2, 1, 1, 2, 2, 1, 1, 2, 1],
        # ]))
        #
        # # We want to set it so that the bottom left coordinate is 0, 0
        # self.state = np.flipud(self.state)
        #
        # self.state = np.int_(np.array([
        #     [0, 1, 1, 2, 2, 2, 0, 1, 0],
        #     [0, 2, 0, 0, 0, 2, 0, 1, 2],
        #     [1, 1, 1, 2, 1, 0, 0, 2, 2],
        #     [1, 1, 2, 2, 2, 1, 0, 1, 2],
        #     [1, 2, 2, 2, 1, 2, 0, 2, 1],
        #     [2, 1, 1, 2, 2, 1, 0, 0, 1],
        # ]))

        self.cluster_history = cluster_history
        if array is None:
            self.grid = np.int_(np.array([
                [1, 1, 1],
                [2, 2, 1]
            ]))
        else:
            self.grid = array
        self.run_simulator()
        self.cluster_count = len(get_clusters(self.grid))

    def bubble_up(self):
        for c in range(0, COL_COUNT):
            col = self.grid[:, c]
            nonz = col[col > 0]
            zeroCount = ROW_COUNT - np.shape(nonz)[0]
            sorted = np.append(nonz, np.zeros(zeroCount))
            self.grid = np.delete(self.grid, c, 1)
            self.grid = np.insert(self.grid, c, sorted, 1)

    def clear_empty_cols(self):
        idx = np.argwhere(np.all(self.grid[..., :] == 0, axis=0))
        column_count = len(idx)
        zeros = np.zeros((ROW_COUNT, column_count), dtype=int)
        self.grid = np.delete(self.grid, idx, axis=1)
        self.grid = np.append(self.grid, zeros, axis=1)

    def run_simulator(self):
        self.bubble_up()
        self.clear_empty_cols()

    def pop(self, cluster):
        for node in cluster:
            self.grid[node.row][node.col] = 0
        self.run_simulator()

    def next_state(self, cluster):
        array = deepcopy(self.grid)
        new_state = WorldState(array, cluster_history=[*self.cluster_history, cluster])
        new_state.pop(cluster)
        return new_state


def get_clusters(grid):
    state_shape = np.shape(grid)
    rows = state_shape[0]
    cols = state_shape[1]

    indices = []
    for i in range(0, rows):
        for j in range(0, cols):
            indices.append(Coordinate(i, j))

    cluster_list = []
    indices = set(indices)
    while indices.__len__() > 0:
        node = list(indices).pop(0)
        cluster = get_cluster(grid, node.row, node.col, grid[node.row][node.col], set())
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
    if row < 0 or row >= ROW_COUNT:
        return {}
    if col < 0 or col >= COL_COUNT:
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


def get_next_state(state, cluster):
    return 4


# x = get_cluster(world.state, 0, 0, 1, set())
# print(x)


# print({Coordinate(0, 0), Coordinate(0, 0)})


class Node:
    def __init__(self, state, parent_node, parent_cluster):
        self.state = state
        self.parent_node = parent_node
        self.parent_cluster = parent_cluster

    def __cmp__(self, other):
        self_clusters = self.state.cluster_count
        other_clusters = other.state.cluster_count
        if self_clusters < other_clusters:
            return -1
        elif self_clusters == other_clusters:
            return 0
        else:
            return 1

    def __lt__(self, other):
        self_clusters = self.state.cluster_count
        other_clusters = other.state.cluster_count
        return self_clusters < other_clusters

    def __eq__(self, other):
        self_clusters = self.state.cluster_count
        other_clusters = other.state.cluster_count
        return self_clusters == other_clusters

    def expand(self):
        clusters = get_clusters(self.state.grid)
        node_map = map(lambda cluster: Node(self.state.next_state(cluster), self, cluster),
                       clusters)
        return list(node_map)


class Action:
    def __init__(self, state, cluster):
        self.state = state
        self.cluster = cluster


#
# class World:
#     def __init__(self, size_row, size_col):
#         self.size_row = size_row
#         self.size_col = size_col
#         self.state = [[0] * size_col for i in range(size_row)]
#         self.points = []
#         for col in range(size_col):
#             for row in range(size_row):
#                 value = random.choices([0, 1], weights=[6, 4], k=1)[0]
#                 self.state[row][col] = value
#                 if value == 1:
#                     self.points.append(Node(row, col, 2))
#
#         # print(self.grid)
#
#     def print(self):
#         for row in self.state:
#             print(f'{" | ".join(map(str, row))}')
#         print("\n")
#
#
# world = World(2, 2)


def eval_function(cluster):
    return cluster.__len__() ** 2


# Search for the shortest path to empty
def greedy_best_first_search(node):
    current_node = node
    frontier = [node]

    while len(frontier) != 0:
        frontier.sort()
        best = frontier.pop(0)
        expansion = best.expand()

        if not best.state.grid.any():
            return best

        frontier.extend(expansion)

    return None


initial_state = WorldState(array=initial_state)
root = Node(initial_state, None, {})
result_node = greedy_best_first_search(root)
# print(result_node.state.grid)

# Print answer
current_node = result_node
path = []
while current_node is not None:
    path.insert(0, current_node)
    current_node = current_node.parent_node

count = 0
for node in path:
    count += 1
    print(f'Step {count}')
    print(np.flipud(node.state.grid))
    print()

# clusters = get_clusters(initial_state.state)
# print(initial_state.state)
# print(clusters)
# initial_state.pop(clusters[0])
# print(initial_state.state)

# print(initial_state.next_state(clusters[1]).state)
