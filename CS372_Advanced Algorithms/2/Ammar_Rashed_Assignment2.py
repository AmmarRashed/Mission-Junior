# Ammar Rashed
# 214200715
# CS 372 Assignment 2

from heapq import heappush, heappop, heapify
from copy import deepcopy
from operator import itemgetter


class Node:
    def __init__(self, state, cost, parent):
        self.state = state
        self.cost = cost
        self.parent = parent

    def __cmp__(self, other):
        return cmp(self.cost, other.cost)

    def __hash__(self):
        return hash(self.state)


class Graph:
    def __init__(self, grid):
        self.grid = grid
        self.neighbors = {}  # {node: [neighbours]}
        self.right_down_neighbors = {}  # {node: [neighbors]}
        self.edge_costs = {}  # {edge: cost}
        self.right_down_edge_cost = {}  # {edge: cost}
        self.nodes = {}  # {(i, j): node}
        self.frontier = []
        self.visited = {}
        self.construct_graph()
        self.checkpoints_start = {}  # [checkpoint: its distance}
        self.checkpoints_end = {}  # [checkpoint: its distance}
        self.start_to_end_path = {}  # deepcopy of self.nodes after a start_to_end solution
        self.end_to_start_path = {}  # deepcopy of self.nodes after a end_to_start solution
        self.init_checkpoints()

    def dijkstra(self, goal, start=(0, 0), start_or_end='start', fullsearch=False):
        self.frontier = []
        self.construct_graph(start=start)
        node = heappop(self.frontier)
        if fullsearch:
            full = True
        else:
            full = node.state == goal
        while len(self.frontier) > 0 or full:
            for neighbor in self.neighbors[node]:
                if neighbor.cost > node.cost + self.edge_costs[(node.state, neighbor.state)]:
                    neighbor.parent = node
                    neighbor.cost = node.cost + self.edge_costs[(node.state, neighbor.state)]
            if start_or_end == 'start' and node.state in self.checkpoints_start:
                self.checkpoints_start[node.state] = node.cost
            elif node.state in self.checkpoints_end:
                self.checkpoints_end[node.state] = node.cost
            heapify(self.frontier)
            try:
                node = heappop(self.frontier)
            except IndexError:
                break
        if start_or_end == 'start':
            self.start_to_end_path = deepcopy(self.nodes)
        else:
            self.end_to_start_path = deepcopy(self.nodes)
        return self.results(start, self.nodes[goal])

    def topological_sort(self, node, graph_stack):  # O(|V|+|E|)
        self.visited[node] = True
        try:
            for neighbor in self.right_down_neighbors[node]:
                if neighbor not in self.visited or not self.visited[neighbor]:
                    self.topological_sort(neighbor, graph_stack)
            graph_stack.append(node)
        except KeyError:
            pass

    def bellman_ford_dag(self, start=(0, 0)):  # Typical Case (|V| + |E|), Worst Case still (|V|*|E|)
        self.visited = {}
        graph_stack = []
        for node in self.nodes:
            if self.grid[node[0]][node[1]] == 'X':
                continue
            if self.nodes[node] not in self.visited or not self.visited[self.nodes[node]]:
                self.topological_sort(self.nodes[node], graph_stack)
        for node in graph_stack[::-1]:
            for neighbor in self.right_down_neighbors[node]:
                edge_cost = self.right_down_edge_cost[(node.state, neighbor.state)]
                if neighbor.cost > node.cost + edge_cost:
                    neighbor.cost = node.cost + edge_cost
                    neighbor.parent = node
        return self.results(start, self.nodes[(len(self.grid) - 1, len(self.grid[0]) - 1)])

    def bellman_ford(self, start=(0, 0)):  # O(|V| * |E|)
        self.frontier = []
        self.construct_graph()
        for i in range(len(self.nodes)-1):
            for edge in self.right_down_edge_cost:
                self.update(edge)
        return self.results(start, self.nodes[(len(self.grid)-1, len(self.grid[0])-1)])

    def update(self, edge):
        u, v = edge
        node_u, node_v = self.nodes[u], self.nodes[v]
        if node_v.cost > node_u.cost + self.right_down_edge_cost[edge]:
            node_v.cost = node_u.cost + self.right_down_edge_cost[edge]
            node_v.parent = node_u

    def results(self, start, goal):
        path = []
        minimum_cost = 0
        while hasattr(goal, 'parent'):
            if goal.state == start:
                break
            path.append(goal.state)
            goal = goal.parent
            try:
                minimum_cost += self.edge_costs[(goal.parent.state, goal.state)]
            except AttributeError:
                pass
        path.append(start)
        return minimum_cost, len(path)-1, path[::-1]

    def construct_graph(self, start=(0, 0)):
        self.neighbors = {}
        self.right_down_neighbors = {}
        for i in range(len(self.grid)):  # row
            for j in range(len(self.grid[0])):  # column
                if i == start[0] and j == start[1]:
                    if (i, j) in self.nodes:
                        self.nodes[(i, j)].cost = 0
                        self.nodes[(i, j)].parent = None
                    else:
                        self.nodes[(i, j)] = Node((i, j), 0, None)
                else:
                    if (i, j) in self.nodes:
                        self.nodes[(i, j)].cost = float('inf')
                        self.nodes[(i, j)].parent = None
                    else:
                        self.nodes[(i, j)] = Node((i, j), float('inf'), None)
                heappush(self.frontier, self.nodes[(i, j)])
        for node in self.nodes.values():
            self.get_neighbors(node)

    def get_neighbors(self, node):
        i, j = node.state
        self.neighbors.setdefault(node, [])
        self.right_down_neighbors.setdefault(node, [])
        # Vertical moves
        try:
            if i == 0:  # only down
                try:
                    x, cost = self.move_down(i, j)
                    self.neighbors[node].append(self.nodes[(x, j)])
                    self.right_down_neighbors[node].append(self.nodes[(x, j)])
                    self.edge_costs[((i, j), (x, j))] = cost
                    self.right_down_edge_cost[((i, j), (x, j))] = cost
                except TypeError:
                    pass

            elif i == len(self.grid)-1:  # only up
                try:
                    x, cost = self.move_up(i, j)
                    self.neighbors[node].append(self.nodes[(x, j)])
                    self.edge_costs[((i, j), (x, j))] = cost
                except TypeError:
                    pass
            else:  # both down and up
                try:
                    x_up, cost_up = self.move_up(i, j)
                    self.neighbors[node].append(self.nodes[(x_up, j)])
                    self.edge_costs[((i, j), (x_up, j))] = cost_up
                except TypeError:
                    pass
                try:
                    x_down, cost_down = self.move_down(i, j)
                    self.neighbors[node].append(self.nodes[(x_down, j)])
                    self.right_down_neighbors[node].append(self.nodes[(x_down, j)])
                    self.edge_costs[((i, j), (x_down, j))] = cost_down
                    self.right_down_edge_cost[((i, j), (x_down, j))] = cost_down
                except TypeError:
                    pass
        except IndexError:
            pass

        # Horizontal moves
        try:
            if j == 0:
                try:  # only right
                    y, cost = self.move_right(i, j)
                    self.neighbors[node].append(self.nodes[(i, y)])
                    self.right_down_neighbors[node].append(self.nodes[(i, y)])
                    self.edge_costs[((i, j), (i, y))] = cost
                    self.right_down_edge_cost[((i, j), (i, y))] = cost
                except TypeError:
                    pass
            elif j == len(self.grid[0])-1:  # only left
                try:
                    y, cost = self.move_left(i, j)
                    self.neighbors[node].append(self.nodes[(i, y)])
                    self.edge_costs[((i, j), (i, y))] = cost
                except TypeError:
                    pass
            else:  # both right and left
                try:
                    y_right, cost_right = self.move_right(i, j)
                    self.neighbors[node].append(self.nodes[(i, y_right)])
                    self.right_down_neighbors[node].append(self.nodes[(i, y_right)])
                    self.edge_costs[((i, j), (i, y_right))] = cost_right
                    self.right_down_edge_cost[((i, j), (i, y_right))] = cost_right
                except TypeError:
                    pass
                try:
                    y_left, cost_left = self.move_left(i, j)
                    self.neighbors[node].append(self.nodes[(i, y_left)])
                    self.edge_costs[((i, j), (i, y_left))] = cost_left
                except TypeError:
                    pass
        except IndexError:
            pass

    def move_down(self, i, j):
        if self.grid[i+1][j] == 'X':
            if self.grid[i+2][j] != 'X':
                return i+2, self.grid[i+2][j]*2
        else:
            return i+1, self.grid[i+1][j]

    def move_up(self, i, j):
        if self.grid[i-1][j] == 'X':
            if self.grid[i-2][j] != 'X' and i - 1 != 0:
                return i - 2, self.grid[i-2][j] * 2
        else:
            return i-1, self.grid[i-1][j]

    def move_right(self, i, j):
        if self.grid[i][j+1] == 'X':
            if self.grid[i][j+2] != 'X':
                return j+2, self.grid[i][j+2] * 2
        else:
            return j+1, self.grid[i][j+1]

    def move_left(self, i, j):
        if self.grid[i][j-1] == 'X':
            if self.grid[i][j-2] != 'X' and j - 1 != 0:
                return j-2, self.grid[i][j-2] * 2
        else:
            return j - 1, self.grid[i][j - 1]

    def init_checkpoints(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] == "Check":
                    self.grid[i][j] = 0
                    self.checkpoints_start[(i, j)] = 0
                    self.checkpoints_end[(i, j)] = 0

    def dijkstra_checkpoint(self):
        ''' We run Dijkstra from start to end which will update the shortest distance between each node and the start,
        then we run Dijkstra from the end to the start so this time we will have the shortest distance between each
        node and the end. Knowing where the checkpoints are, we sum the shortest distances between each checkpoint
        to the start and to the end. The checkpoint with the smallest distance has to be included in the path, so
        get the path from the start to this checkpoint (using result() method) and from the end to the checkpoint,
        merge the two paths and sum their number of steps and path total cost, and Voiala we have the answer!
        In order to differentiate between the path from start to end and vice versa, we make two different dicts,
        one for each, and they are simply deep copies of self.nodes which maps each state to its node object.
        This way we can easily rebuild any path by traversing through the parents of the goal node all the way
        up to the start node. That means once we know the optimal checkpoint to pass through, we do not need to
        go and run Dijkstra two more times to find the paths from the start and the checkpoint and to the end, since
        we already computed them the first time. That makes the total complexity of this algorithm:
                        O((V+E)logV)
        The only problem here though is the heapify function that we need to run every time we update the cost of a node
        in the frontier heap priority queue, whose complexity is linear.
        '''
        self.dijkstra(goal=(len(self.grid)-1, len(self.grid[0])-1), start=(0, 0), fullsearch=True)  # from top-left to bottom-right
        self.dijkstra(goal=(0, 0), start=(len(self.grid)-1, len(self.grid[0])-1), start_or_end='end', fullsearch=True)  # from bottom-right to top-left
        checkpoints = {}  # {checkpoint:shortest distance from start + shortest distance from the end}
        for checkpoint in self.checkpoints_start:
            checkpoints[checkpoint] = self.checkpoints_start[checkpoint]+self.checkpoints_end[checkpoint]
        optimal_checkpoint = min(checkpoints, key=itemgetter(1))
        path_start_to_checkpoint = self.results(start=(0, 0), goal=self.start_to_end_path[optimal_checkpoint])
        path_end_to_checkpoint = self.results(start=(len(self.grid)-1, len(self.grid[0])-1), goal=self.end_to_start_path[optimal_checkpoint])
        path = path_start_to_checkpoint[-1][:-1]+path_end_to_checkpoint[-1][::-1]
        cost = path_start_to_checkpoint[0]+path_end_to_checkpoint[0]
        steps_num = path_start_to_checkpoint[1]+path_end_to_checkpoint[1]
        return cost, steps_num, path


def find_shortest_path(g):
    graph = Graph(g)
    minimum_cost, steps, path = graph.dijkstra(goal=(len(g)-1, len(g[0])-1))
    return minimum_cost, steps, path


def find_shortest_path_with_negative_costs(g):
    graph = Graph(g)
    return graph.bellman_ford_dag()  # minimum_cost, steps, path
    # return graph.bellman_ford()

def find_shortest_path_with_checkpoint(g):
    graph = Graph(g)
    return graph.dijkstra_checkpoint()
# print find_shortest_path([[0, 'X', 1, 4, 9, 'X'],
#                           [7, 7, 4, 'X', 4, 8],
#                           [3, 'X', 3, 2, 'X', 4],
#                           [10, 2, 5, 'X', 3, 0]])
# print find_shortest_path_with_negative_costs([[0, 'X', 5, -3, 6, 'X'],
#                                               [6, -4, 5, 'X', 3, -8],
#                                               [4, 'X', 3, -7, 'X', 5],
#                                               [10, 2, -2, 'X', 6, 0]])
#
# print find_shortest_path_with_checkpoint([[0, 6, 'Check', 4, 9, 3],
#                                           [5, 7, 4, 6, 'Check', 3],
#                                           [3, 6, 'Check', 5, 8, 4],
#                                           [10, 2, 5, 4, 3, 0]])
