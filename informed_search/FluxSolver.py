# Model
# Generate Grid

import numpy as np
from copy import deepcopy

# Create World State
# Find all clusters

# Goal Formulation

initial_state = np.int_(np.array([
    [1, 1, 1, 1, 1, 2, 1, 1, 1],
    [1, 2, 2, 1, 1, 2, 2, 1, 2],
    [1, 1, 1, 1, 2, 1, 1, 1, 2],
    [2, 1, 2, 2, 1, 1, 2, 2, 2],
    [2, 2, 1, 2, 1, 1, 2, 2, 1],
    [1, 1, 2, 2, 1, 2, 2, 1, 2],
]))
initial_state = np.flipud(initial_state)
initial_shape = np.shape(initial_state)
ROW_COUNT = initial_shape[0]
COL_COUNT = initial_shape[1]


class WorldState:
    def __init__(self, array=None):
        self.grid = array
        self.run_simulator()

    def get_moves(self):
        return get_moves(self.grid)

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
        new_state = WorldState(array)
        new_state.pop(cluster)
        return new_state

    def zero_count(self):
        return np.count_nonzero(self.grid == 0)


def get_moves(grid):
    state_shape = np.shape(grid)
    rows = state_shape[0]
    cols = state_shape[1]

    indices = []
    for i in range(0, rows):
        for j in range(0, cols):
            indices.append(Coordinate(i, j))

    cluster_list = []
    indices = set(indices)
    while len(indices) > 0:
        coord = indices.pop()
        value = grid[coord.row][coord.col]
        if value == 0:
            continue
        current_cluster = get_cluster(grid, coord.row, coord.col, value, set())
        cluster_list.append(current_cluster)
        indices.difference_update(current_cluster)

    return cluster_list


class Coordinate:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __hash__(self):
        return hash(f'{self.row},{self.col}')

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __lt__(self, other):
        return self.row < other.row or (self.row == other.row and self.col < other.col)

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


class Node:
    def __init__(self, state, parent_node, parent_cluster):
        self.state = state
        self.parent_node = parent_node
        self.parent_cluster = parent_cluster

    def __lt__(self, other):
        self_clusters = self.state.zero_count()
        other_clusters = other.state.zero_count()
        return self_clusters > other_clusters

    def expand(self):
        move_list = self.state.get_moves()
        expansion_list = []
        for move in move_list:
            if len(move) <= 1:
                continue
            node = Node(self.state.next_state(move), self, move)
            expansion_list.append(node)
        return expansion_list


# Search for the shortest path to empty
def valid_moves(node_list):
    moves = []
    for node in node_list:
        if node.state.move_count > 1:
            moves.append(node)
    return moves


def greedy_best_first_search(node):
    frontier = [node]

    while len(frontier) != 0:
        frontier.sort()
        best = frontier.pop(0)
        expansion_nodes = best.expand()
        if not best.state.grid.any():
            return best

        frontier.extend(expansion_nodes)

    return None


initial_state = WorldState(array=initial_state)
root = Node(initial_state, None, {})
result_node = greedy_best_first_search(root)

# Print answer
if result_node is None:
    print(f'No solution found')
    exit(0)

current_node = result_node
path = []
while current_node is not None:
    path.insert(0, current_node)
    current_node = current_node.parent_node

count = 0
for node in path:
    count += 1
    print(f'Step {count}')
    print(f'Cluster:')
    cluster = list(node.parent_cluster)
    cluster.sort()
    print(cluster)
    print(np.flipud(node.state.grid))
    print()
