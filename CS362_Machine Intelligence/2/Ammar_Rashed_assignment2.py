from copy import deepcopy
import visualise
def get_empty_blocks_indices(state):
    """
    Finding the x,y index of each empty block
    :param state: 2-dimensional array representing current block world config
    :return: indices: list of tuples [(x1,y1), (x2,y2)] representing the coordinates of each empty block
    """
    indices = []
    for row in range(len(state)):
        for column in range(len(state[row])):
            if state[row][column] == 0:
                indices.append((row, column))
    return indices

def right_blue(state, r, c):
    temp = state[r][c+1]
    state[r][c+1] = 0
    state[r][c] = temp

    return state

def left_blue(state, r, c):
    temp = state[r][c - 1]
    state[r][c - 1] = 0
    state[r][c] = temp
    return state

def right_green(state, r, c1, c2):
    temp1 = state[r][c1 + 1]
    state[r][c1 + 1] = 0
    state[r][c1] = temp1

    temp2 = state[r][c1 + 2]
    state[r][c2 + 2] = 0
    state[r][c2] = temp2
    return state

def left_green(state, r, c1, c2):
    temp1 = state[r][c1 - 1]
    state[r][c1 - 1] = 0
    state[r][c1] = temp1

    temp2 = state[r][c1 - 2]
    state[r][c2 - 2] = 0
    state[r][c2] = temp2
    return state


def check_updown_green(state, direction, x1, x2, y):
    """ Checking if there are two vertically aligned 1x1 green blocks above/below
     of the two vertically aligned empty 1x1 block.
    direction means a row above or below,
     1 means above
    -1 means below"""
    if direction == 1 and min(x1,x2) < 2: return False  # The empty tiles are already at the top of the block config
    if direction == -1 and max(x1,x2) >= len(state)-2: return False  # ,, ,, ,, bottom of the block config

    if direction > 0:  # Two 1x1 blocks above the empty tiles are green?
        x = min(x1, x2)
        return state[x - direction][y] == 2 and state[x - 2*direction][y] == 2
    else:  # Two 1x1 blocks below the empty tiles are green?
        x = max(x1, x2)
        return state[x - direction][y] == 2 and state[x - 2*direction][y] == 2

def check_vertical_red(state, direction, x1, x2, y):
    """ Checking if there are two vertically aligned 1x1 red blocks to the right/left
        of the two vertically aligned empty 1x1 block.
       direction means a row above or below,
        1 means right
       -1 means left"""
    if direction == 1 and y >= len(state[0])-2: return False  # The empty tiles are already at the right-most
    if direction == -1 and y < 2: return False  # ,, ,, ,, left-most of the block config
    return state[x1][y + direction] == 4 and state[x2][y + direction] == 4

def check_vertical_green(state, direction, x1, x2, y):
    """ Checking if there are two vertically aligned 1x1 green blocks to the right/left
        of the two vertically aligned empty 1x1 block.
       direction means a row above or below,
        1 means right
       -1 means left"""
    if direction == 1 and y == len(state[0])-1: return False  # The empty tiles are already at the right-most
    if direction == -1 and y == 0: return False  # ,, ,, ,, left-most of the block config
    return state[x1][y + direction] == 2 and state[x2][y + direction] == 2



def check_horizontal_green(state, direction, x, y1, y2):
    """ Checking if there are two horizontally aligned 1x1 green blocks below/above
    the two horizontally aligned empty 1x1 blocks.
    direction means a row above or below,
     1 means above
    -1 means below"""
    if direction == 1 and x == 0: return False  # The empty tiles are already at the top of the block config
    if direction == -1 and x == len(state)-1: return False  # ,, ,, ,, bottom of the block config
    return state[x-direction][y1] == 2 and state[x - direction][y2] == 2


def check_horizontal_red(state, direction, x, y1, y2):
    """ Check if there are two horizontally aligned 1x1 red blocks below/above
        the two horizontally aligned empty 1x1 blocks.
        direction means a row above or below,
     1 means above
    -1 means below"""
    if direction == 1 and x < 2: return False  # The empty tiles are already at the top of the block config
    if direction == -1 and x >= len(state) - 2: return False  # ,, ,, ,, bottom of the block config
    return state[x-direction][y1] == 4 and state[x - direction][y2] == 4


def check_up_down_blue(state, direction, x, y):
    """ direction means a row above or below,
     1 means above
    -1 means below"""
    if direction == 1 and x == 0: return False  # The empty tiles are already at the top of the block config
    if direction == -1 and x == len(state) - 1: return False  # ,, ,, ,, bottom of the block config
    return state[x-direction][y] == 1


def check_right_left_blue(state, direction, x, y):
    """ direction means a row above or below,
     1 means right
    -1 means left"""
    if direction == 1 and y == len(state[0])-1: return False  # The empty tiles are already at the right-most
    if direction == -1 and y == 0: return False  # ,, ,, ,, left-most of the block config
    return state[x][y + direction] == 1


def check_right_left_green(state, direction, x, y1, y2):
    """ direction means a column right or left,
     1 means right
    -1 means left"""
    if direction == 1:
        y = max(y1, y2)
        if y >= len(state[0])-2: return False  # The empty tiles are already at the right-most
    if direction == -1:
        y = min(y1, y2)
        if y <= 1: return False  # ,, ,, ,, left-most of the block config
    return state[x][y + direction] == 2 and state[x][y + direction*2] == 2



def vertical_move_horizontal_green_horizontal_blanks(state, direction, x, y1, y2):
    """ Swap two horizontally aligned 1x1 empty blocks with other two horizontally aligned 1x1 green blocks below/above
        them, direction means a row above or below,
        1 means above
       -1 means below"""
    temp1 = state[x - direction][y1]
    temp2 = state[x - direction][y2]
    state[x - direction][y1] = state[x][y1]
    state[x - direction][y2] = state[x][y2]
    state[x][y1] = temp1
    state[x][y2] = temp2
    return state


def move_vertical_green(state, direction, x1, x2, y):
    """ Swap two vertically aligned empty 1x1 blocks with other two vertically aligned 1x1 green blocks
        in the same column up or down direction means a row above or below,
        1 means above
       -1 means below"""
    direction *= 2
    temp1 = state[x1 - direction][y]
    temp2 = state[x2 - direction][y]
    state[x1 - direction][y] = state[x1][y]
    state[x2 - direction][y] = state[x2][y]
    state[x1][y] = temp1
    state[x2][y] = temp2
    return state


def swap_right_left_green_horizontal_empty_tiles(state, direction ,x, y1, y2):
    """ Swapping two horizontally aligned 1x1 green blocks with the two vertically aligned empty 1x1 blocks
        to the right/left of them direction means a row above or below,
        1 means right
       -1 means left"""
    temp1 = state[x][y1 + direction*2]
    temp2 = state[x][y2 + direction*2]
    state[x][y1 + 2*direction] = state[x][y1]
    state[x][y2 + 2*direction] = state[x][y2]
    state[x][y1] = temp1
    state[x][y2] = temp2
    return state


def swap_right_left_green(state, direction ,x1, x2, y):
    """ Swapping two vertically aligned 1x1 green blocks with the two vertically aligned empty 1x1 blocks
        to the right/left of them direction means a row above or below,
        1 means right
       -1 means left"""
    temp1 = state[x1][y + direction]
    temp2 = state[x2][y + direction]
    state[x1][y + direction] = state[x1][y]
    state[x2][y + direction] = state[x2][y]
    state[x1][y] = temp1
    state[x2][y] = temp2
    return state


def swap_right_left_blue(state, direction, x, y):
    """ Swapping 1x1 blue block with the empty 1x1 empty tile on the lef/right
        direction means a row above or below,
        1 means right
       -1 means left"""
    temp = state[x][y + direction]
    state[x][y + direction] = state[x][y]
    state[x][y] = temp
    return state


def move_vertical_blue(state, direction, x, y):
    """ direction means a row above or below,
        1 means above
       -1 means below"""

    temp = state[x - direction][y]
    state[x - direction][y] = state[x][y]
    state[x][y] = temp
    return state


def move_vertical_red(state, direction, x, y1, y2):
    """ direction means a row above or below,
        1 means above
       -1 means below"""
    temp1 = state[x - direction*2][y1]
    temp2 = state[x - direction*2][y2]
    state[x - direction*2][y1] = state[x][y1]
    state[x - direction*2][y2] = state[x][y2]
    state[x][y1] = temp1
    state[x][y2] = temp2
    return state


def redblock_xy_distance_from_goal(state):
    """ Finding the x,y distance of the red block from the bottom left corner
        Parameters
        ---------
        state: 2-dimensional array representing current block world config
        Returns
        -------
        distance: (x,y) tuple of the x,y coordinates"""
    for row in range(len(state)):
        if 4 in state[row]:  # Finds the 1st appearance of a red block in a row
            # The state[row] is the y index from the left, and row+1 is the x index from the top
            return state[row].index(4), (len(state)-1 - (row+1))  # len(state) is e.g. 4, but the last index is 4-1=3


def get_successors(initial_state):
    """Successor function for the block world puzzle
       Parameters
       ----------
       initial_state: 2-dimensional array representing initial block world config
       Returns
       -------
       expanded_states: type - list of 2D arrrays
    """
    expanded_states = []
    # Initializing the location of the empty blocks
    indices = get_empty_blocks_indices(initial_state)
    x1, y1 = indices[0]
    x2, y2 = indices[1]

    old_state = deepcopy(initial_state)  # Passing-by-value to a clone variable to avoid changing the input state

    # _______________Check the alignment of the two empty blocks_______________
    # Horizontal alignment
    for direction in [1, -1]:  # one for moving the empty tile up/right, and the other for moving down/left.
        if x1 == x2 and max(y1,y2)== min(y1,y2)+1:
            # >>>>> horizontal swaps with blue blocks
            for y in [y1, y2]:
                if check_right_left_blue(old_state, direction, x1, y):
                    swap_right_left_blue(old_state, direction, x1, y)
                old_state = deepcopy(initial_state)  # Resetting the clone variable
            # >>>>> horizontal swaps with green blocks
            if check_right_left_green(old_state, direction, x1, y1, y2):
                swap_right_left_green_horizontal_empty_tiles(old_state, direction, x1, y1, y2)
            old_state = deepcopy(initial_state)  # Resetting the clone variable
            # """From here on, I started refactoring my code :)"""
            # >>>>> vertical swaps
            #  >>>>> vertical swaps with green or red blocks
            if check_horizontal_green(old_state, direction= direction, x=x1, y1=y1, y2=y2):
                # There are two horizontally aligned green 1x1 blocks below/above
                state = vertical_move_horizontal_green_horizontal_blanks(old_state, direction, x1, y1, y2)
                if state not in expanded_states: expanded_states.append(state)
            elif check_horizontal_red(old_state, direction = direction, x=x1, y1=y1, y2=y2):
                # There are two horizontally aligned red 1x1 blocks below/above,
                # which means there is a 4x4 red block below/above
                state = move_vertical_red(old_state, direction, x1, y1, y2)
                if state not in expanded_states: expanded_states.append(state)

            # >>>>> vertical swaps with blue blocks
            # First empty tile
            elif check_up_down_blue(old_state, direction, x1, y1):
                state = move_vertical_blue(old_state, direction, x1, y1)
                if state not in expanded_states: expanded_states.append(state)
            old_state = deepcopy(initial_state)  # Resetting the clone variable
            # Second empty tile
            if check_up_down_blue(old_state, direction, x1, y2):
                state = move_vertical_blue(old_state, direction, x1, y2)
                if state not in expanded_states: expanded_states.append(state)
            old_state = deepcopy(initial_state)  # Resetting the clone variable

        # Vertical alignment
        elif y1 == y2 and max(x1, x2)== min(x1, x2)+1:
            # >>>>> vertical swaps with green blocks
            if check_updown_green(old_state, direction, x1, x2, y1):
                state = move_vertical_green(old_state, direction, x1, x2, y1)
                if state not in expanded_states: expanded_states.append(state)
            # >>>>> vertical swaps with blue blocks
            # First empty tile
            elif check_up_down_blue(old_state, direction, x1, y1):
                state = move_vertical_blue(old_state, direction, x1, y1)
                if state not in expanded_states: expanded_states.append(state)
            old_state = deepcopy(initial_state)  # Resetting the clone variable
            # Second empty tile
            if check_up_down_blue(old_state, direction, x2, y1):
                state = move_vertical_blue(old_state, direction, x1, y2)
                if state not in expanded_states: expanded_states.append(state)

            old_state = deepcopy(initial_state)  # Resetting the clone variable

            # >>>>> horizontal swaps with green blocks
            if check_vertical_green(old_state, direction, x1, x2, y1):
                state = swap_right_left_green(old_state, direction, x1, x2, y1)
                if state not in expanded_states: expanded_states.append(state)

            # >>>>> horizontal swaps with red blocks
            elif check_vertical_red(old_state, direction, x1, x2, y1):
                state = swap_right_left_green(old_state, direction*2, x1, x2, y1)
                if state not in expanded_states: expanded_states.append(state)

                # >>>>> horizontal swaps with blue blocks
            elif check_right_left_blue(old_state, direction, x1, y1):
                state = swap_right_left_blue(old_state, direction, x1, y1)
                if state not in expanded_states: expanded_states.append(state)
            old_state = deepcopy(initial_state)  # Resetting the clone variable
            if check_right_left_blue(old_state, direction, x2, y2):
                state = swap_right_left_blue(old_state, direction, x2, y2)
                if state not in expanded_states: expanded_states.append(state)
            old_state = deepcopy(initial_state)  # Resetting the clone variable
        else:
            # >>>>> horizontal swaps (possible only with blue blocks)
            # First empty tile
            if check_right_left_blue(old_state, direction, x1, y1):
                state = swap_right_left_blue(old_state, direction, x1, y1)
                if state not in expanded_states: expanded_states.append(state)
            old_state = deepcopy(initial_state)  # Resetting the clone variable
            # Second empty tile
            if check_right_left_blue(old_state, direction, x2, y2):
                state = swap_right_left_blue(old_state, direction, x2, y2)
                if state not in expanded_states: expanded_states.append(state)
            old_state = deepcopy(initial_state)  # Resetting the clone variable

            # >>>>> vertical swaps (possible only with blue blocks)
            # First empty tile
            if check_up_down_blue(old_state, direction, x1, y1):
                state = move_vertical_blue(old_state, direction, x1, y1)
                if state not in expanded_states: expanded_states.append(state)
            old_state = deepcopy(initial_state)  # Resetting the clone variable
            # Second empty tile
            if check_up_down_blue(old_state, direction, x2, y2):
                state = move_vertical_blue(old_state, direction, x2, y2)
                if state not in expanded_states: expanded_states.append(state)
            old_state = deepcopy(initial_state)  # Resetting the clone variable
    return expanded_states


def is_goal(state):
    """ Checks if the red block is at the left bottom corner of the block config
        :returns
        boolean True or False for the aforementioned condition
    """
    # return state == [[1, 2, 1, 1], [1, 2, 4, 4], [2, 2, 4, 4], [1, 0, 1, 0]]
    return state[-1][0] == 4 and state[-1][1] == 4


class Node:
    def __init__(self, state, cost, parent):
        self.state = state
        self.cost = cost
        self.parent = parent

def in_explored(state, explored):
    for node in explored:
        if node.state == state: return node
    return None


def uniform_cost_search(initial_state, heuristic=lambda state : 1):
    """Finds the path taken by the uniform cost search algorithm
       Parameters
       ----------
       initial_state: 2-dimensional array representing initial block world config
       heuristic: heuristic function, for UCS (default) it is always 1
       Returns
       -------
       path: (sequence of states to solve the block world) type - list of 2D arrays
    """
    # frontier = The successors of the explored states that have not been explored yet
    initial_state_cost = heuristic(initial_state)
    if initial_state_cost == 1:
        frontier = [Node(initial_state, 0, None)]  # [Nodes], initialized with root state with cost 0
    else:
        frontier = [Node(initial_state, initial_state_cost, None)]  # [Nodes], initialized with root state with cost 0
    explored = []  # Explored states [Nodes] list of Node objects
    visited = []  # Visited states [states]  list of 2-d lists
    while len(frontier) > 0:
        frontier.sort(key= lambda x: x.cost)
        min_cost_node = frontier.pop(0)
        if is_goal(min_cost_node.state):  # Goal found
            explored.append(min_cost_node)
            break
        explored_node = in_explored(min_cost_node.state, explored)
        if min_cost_node.state in visited:
            # Updating the min cost of that node
            if explored_node != None and new_cost < explored_node.cost:  # It has been visited before
                explored_node.cost = new_cost
            continue
        visited.append(min_cost_node.state)
        explored.append(min_cost_node)
        # else:  # that node was not in the explored nodes
        for successor in get_successors(min_cost_node.state):
            new_cost = (heuristic(successor) + min_cost_node.cost)
            # if successor not in visited and explored_node==None:
            frontier.append(Node(successor, new_cost, min_cost_node))

    # Rebuilding the path
    path = []
    node = explored[-1]  # This should be the goal
    try:
        assert is_goal(node.state)
    except AssertionError:
        print "Could not reach the goal"
    while node.parent != None:
        path.append(node.state)
        node = node.parent
    path.append(initial_state)
    return path[::-1]

def a_star_heuristic(state):
    """Euclidean distance heuristic for a star algorithm
       Parameters
       ----------
       state: 2-dimensional array representing block world state
       Returns
       -------
       Euclidean distance (type- float) as defined in the assignment description
    """
    x, y = redblock_xy_distance_from_goal(state)
    return (x**2+y**2)**0.5


def a_star_search(initial_state):
    """Finds the path taken by the a star search algorithm
       Parameters
       ----------
       initial_state: 2-dimensional array representing initial block world config
       Returns
       -------
       path: (sequence of states to solve the block world) type-list of 2D arrays
    """
    return uniform_cost_search(initial_state, heuristic=a_star_heuristic)

if __name__ == "__main__":
    visualise.start_simulation(uniform_cost_search([ [1,2,1,1], [1,2,4,4], [2,2,4,4], [1,1,0,0] ]))
# print get_successors([ [1,2,1,1], [1,2,4,4], [2,2,4,4], [1,1,0,0] ])
# print uniform_cost_search([ [1,2,1,1], [1,2,4,4], [2,2,4,4], [1,1,0,0] ])
# print a_star_search([ [1,2,1,1], [1,2,4,4], [2,2,4,4], [1,1,0,0] ])