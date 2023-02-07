import random


class Node:
    def __init__(self, height, depth, branching):
        self.height = height
        self.routes = []
        self.cost = random.randint(0, 100)
        self.is_goal = random.choices([True, False], weights=[5, 95], k=1)[0]
        if depth > 0:
            for i in range(0, branching):
                new_node = Node(height + 1, depth - 1, random.randint(0, branching))
                route = Route(new_node, random.randint(0, 100))
                self.routes.append(route)

    def __str__(self):
        return f'Height: {self.height}\nCost: {self.cost}\nIs Goal: {self.is_goal}'

    def display_string(self):
        return f'Height: {self.height}\nCost: {self.cost}\nIs Goal: {self.is_goal}'


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
        print(f"Node {count}:")
        print(node)
        print("\n")

        if node.is_goal:
            return node

        for child in node.routes:
            frontier.append(child.node)
    return None


n = Node(0, 4, 3)
goal = depth_first_search(n)
if goal is None:
    print("No goal found")
else:
    print(f'Found goal:\n{goal}')
