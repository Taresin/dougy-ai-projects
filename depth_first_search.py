import random
import time
import names


class Node:
    node_count = 0

    def __init__(self, parent, height, depth, branching):
        Node.node_count += 1
        self.name = names.get_first_name()
        self.parent = parent
        self.height = height
        self.routes = []
        self.cost = random.randint(0, 100)
        self.is_goal = random.choices([True, False], weights=[5, 95], k=1)[0]

        # print(f'Created Node:\n{self}')
        if depth > 0:
            for i in range(0, branching):
                new_node = Node(self, height + 1, depth - 1, random.randint(0, branching))
                route = Route(new_node, random.randint(0, 100))
                self.routes.append(route)

    def __str__(self):
        tab = ""
        return f'{tab}Name:{self.name}\n{tab}Height: {self.height}\n{tab}Cost: {self.cost}\n{tab}Is Goal: {self.is_goal}'
        # return f'Height: {self.height}\nCost: {self.cost}\nIs Goal: {self.is_goal}'

    def display_string(self, number):
        tab = self.height * "\t"
        return f'{tab} {self.height} - Node {number} ({self.name}) {"(Goal)" if self.is_goal else ""}'


class Route:
    def __init__(self, node, cost):
        self.node = node
        self.cost = cost


def depth_first_search(starting_node):
    count = 0
    frontier = [starting_node]
    while frontier.__len__() > 0:
        count += 1
        node = frontier.pop()
        # print(f"Node {count}:")
        print(node.display_string(count))
        time.sleep(0.5)

        if node.is_goal:
            print(f'Searched through {count} of {Node.node_count} nodes.')
            return node

        for child in node.routes:
            frontier.append(child.node)
    print(f'Searched through all nodes.')
    return None


n = Node(None, 0, 4, 3)
print(f'Nodes created: {Node.node_count}')
goal = depth_first_search(n)

if goal is None:
    print("No goal found")
else:
    print()
    path = [goal]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    print("The path to the goal is this:")
    for node in path:
        time.sleep(1)
        print(f"{node}\n")
