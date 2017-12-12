from copy import deepcopy  # For pass-by-value simulation
states_lineage = []  # [(state, parent), ...]

def getEmptyTileIndex(state):
    for row in state:
        if 0 in row:
            index = (state.index(row), row.index(0))
            return index
"""Defining moves
    r for row and c for column"""
def up(state):
    r, c = getEmptyTileIndex(state)
    temp = state[r-1][c]
    state[r - 1][c] = 0
    state[r][c] = temp

    return state

def down(state):
    r, c = getEmptyTileIndex(state)
    temp = state[r + 1][c]
    state[r + 1][c] = 0
    state[r][c] = temp

    return state

def right(state):
    r, c = getEmptyTileIndex(state)
    temp = state[r][c+1]
    state[r][c+1] = 0
    state[r][c] = temp

    return state

def left(state):
    r, c = getEmptyTileIndex(state)
    temp = state[r][c - 1]
    state[r][c - 1] = 0
    state[r][c] = temp
    return state

root = []
def expand(initial_state):
    """Successor function for the 8-puzzle variant
       Parameters
       ----------
       initial_state: 2-dimensional array representing initial 8-puzzle config
       Returns
       -------
       expanded_states: type- list of 2D arrrays
       """
    expanded_states = []
    r, c = getEmptyTileIndex(initial_state)
    old_state = deepcopy(initial_state)  # Passing-by-value to a clone variable to avoid changing the input state

    # Horizontal moves
    if c == 0:
        # The empty tile can be swapped with the tile on its right
        expanded_states.append(right(old_state))
    elif c == 1:
        # The empty tile can be swapped with the tile on its left or right
        expanded_states.append(right(old_state))
        old_state = deepcopy(initial_state)  # Resetting the clone variable
        expanded_states.append(left(old_state))
    elif c == 2:
        # The empty tile can be swapped with the tile on its left
        expanded_states.append(left(old_state))
    old_state = deepcopy(initial_state) # Resetting the clone variable
    # Vertical moves
    if r == 0:
        # The empty tile can be swapped with the tile beneath it
        expanded_states.append(down(old_state))
    elif r == 1:
        # The empty tile can be swapped with the tile beneath or above it
        expanded_states.append(up(old_state))
        old_state = deepcopy(initial_state) # Resetting the clone variable
        expanded_states.append(down(old_state))
    elif r == 2:
        # The empty tile can be swapped with the tile above it
        expanded_states.append(up(old_state))
    old_state = deepcopy(initial_state)  # Resetting the clone variable
    # Keep record of the states lineage
    for state in expanded_states:
        states_lineage.append((state, old_state))
    return expanded_states

def isGoal(state):
    r, c = getEmptyTileIndex(state)
    return r == 1 and c == 1

visited_states = []
path = []
def build_path(goal_state):
    global path
    while True:
        if goal_state == root:
            path.append(goal_state)
            break
        for state, parent in states_lineage:
            if state == goal_state:
                path.append(state)
                goal_state = parent
                break

def print_state(state):
    for row in state:
        for column in row:
            print column,
        print
    print "_"*10

def graph_search(initial_state):
    """Defines the path taken by the breadth-first search algorithm
       Parameters
       ----------
       initial_state: 2-dimensional array representing initial 8-puzzle config
       Returns
       -------
       path: (sequence of states to solve the 8-puzzle variant)type-list of 2D arrays
       """
    global root, path
    root = initial_state
    # Defining the base condition
    if isGoal(initial_state):
        return "found \n", initial_state
    visited_states.append(initial_state)

    lst_layer = expand(initial_state)
    for state in lst_layer:  # Going horizontally through the first layer
        if isGoal(state):
            build_path(state)
            return path[::-1]
        visited_states.append(state)

    while True:  # Iterative solution  (memory efficient)
        current_layer = []
        for parent_state in lst_layer:
            for child_state in expand(parent_state):
                if not(child_state in visited_states):
                    if isGoal(child_state):
                        build_path(child_state)
                        return path[::-1]
                    current_layer.append(child_state)  # Populating next generation layer
                    visited_states.append(child_state)
        lst_layer = current_layer


# for state in expand([[5, 4, 0], [6,1,8], [7,3,2]]):
#     print_state(state)
for state in graph_search([[5, 4, 0], [6,1,8], [7,3,2]]):
    print_state(state)
