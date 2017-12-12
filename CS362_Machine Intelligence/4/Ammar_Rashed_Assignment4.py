import operator, copy

reward = {0: 0, 1: -5, 2: -10, 3: 30}


def compute_v_values(grid_config):
    # Initializing the v-values for all the states
    v_values = {}
    output_v_values = {}
    for i in range(len(grid_config)):
        for j in range(len(grid_config[0])):
            v_values[(i, j)] = 0
    # Calculating each V_value gained by taking each action, and choosing the maximum
    for state in v_values:
        if grid_config[len(grid_config) - state[0] - 1][state[1]] == 3:
            output_v_values[state] = 30
            continue
        v_north = north(state, v_values, grid_config)
        v_east = east(state,   v_values, grid_config)
        v_south = south(state, v_values, grid_config)
        v_west = west(state,   v_values, grid_config)
        output_v_values[state] = max(v_east, v_west, v_north, v_south)
    return sorted(output_v_values.items())


def noise_vertical(state, v_values, grid_config):
    global reward
    x, y = state
    v = 0
    # Going north
    if x == len(grid_config) - 1:  # Top edge
        v += 0.15 * (v_values[state] + reward[grid_config[len(grid_config) - x - 1][y]])
    else:
        v += 0.15 * (v_values[(x + 1, y)] + reward[grid_config[len(grid_config) - x - 2][y]])

    # Going south
    if x == 0:  # Bottom edge
        v += 0.15 * (v_values[state] + reward[grid_config[len(grid_config) - x - 1][y]])
    else:
        v += 0.15 * (v_values[state] + reward[grid_config[len(grid_config) - x][y]])
    return v


def noise_horizontal(state, v_values, grid_config):
    global reward
    x, y = state
    v = 0
    # Going east
    if y == len(grid_config[0]) - 1:  # east edge
        v += 0.15 * (v_values[state] + reward[grid_config[len(grid_config) - x - 1][y]])
    else:
        v += 0.15 * (v_values[(x, y + 1)] + reward[grid_config[len(grid_config) - x - 1][y + 1]])
    # Going west
    if y == 0:  # west edge
        v += 0.15 * (v_values[state] + reward[grid_config[len(grid_config) - x - 1][y]])
    else:
        v += 0.15 * (v_values[(x, y - 1)] + reward[grid_config[len(grid_config) - x - 1][y - 1]])
    return v


def east(state, v_values, grid_config):
    global reward
    x, y = state
    v = 0
    # Going east
    if y == len(grid_config[0]) - 1:  # east edge
        v += 0.7 * (v_values[state] + reward[grid_config[len(grid_config) - x - 1][y]])
    else:
        v += 0.7 * (v_values[(x, y+1)] + reward[grid_config[len(grid_config) - x - 1][y+1]])
    v += noise_vertical(state, v_values, grid_config)
    return v - 1


def west(state, v_values, grid_config):
    global reward
    x, y = state
    v = 0
    # Going west
    if y == 0:  # west edge
        v += 0.7 * (v_values[state] + reward[grid_config[len(grid_config) - x - 1][y]])
    else:
        v += 0.7 * (v_values[(x, y - 1)] + reward[grid_config[len(grid_config) - x - 1][y - 1]])
    v += noise_vertical(state, v_values, grid_config)
    return v - 1


def north(state, v_values, grid_config):
    global reward
    x, y = state
    v = 0
    # Going north
    if x == len(grid_config) - 1:  # Top edge
        v += 0.7 * (v_values[state] + reward[grid_config[len(grid_config) - x - 1][y]])
    else:
        v += 0.7 * (v_values[(x + 1, y)] + reward[grid_config[len(grid_config) - x - 2][y]])
    v += noise_horizontal(state, v_values, grid_config)
    return v - 1


def south(state, v_values, grid_config):
    global reward
    x, y = state
    v = 0
    # Going south
    if x == 0:  # Bottom edge
        v += 0.7 * (v_values[state] + reward[grid_config[len(grid_config) - x - 1][y]])
    else:
        v += 0.7 * (v_values[state] + reward[grid_config[len(grid_config) - x][y]])
    v += noise_horizontal(state, v_values, grid_config)
    return v - 1


def modified_v_value_computer(grid_config, v_values):
    # Calculating each V_value gained by taking each action, and choosing the maximum
    optimal_plicy = {}
    output_v_values = {}
    for state in v_values:
        if grid_config[len(grid_config) - state[0] - 1][state[1]] == 3:
            optimal_plicy[state] = 4
            output_v_values[state] = 30
            continue
        v_east = east(state,   v_values, grid_config)
        v_west = west(state,   v_values, grid_config)
        v_north = north(state, v_values, grid_config)
        v_south = south(state, v_values, grid_config)
        vs = [v_north, v_east, v_south, v_west]
        action, v_value= max(enumerate(vs), key=operator.itemgetter(1))
        output_v_values[state] = v_value
        optimal_plicy[state] = action
    return output_v_values, optimal_plicy


def get_optimal_policy(grid_config):
    # Initializing the v-values for all the states
    v_values = {}
    optimal_policy = {}
    for i in range(len(grid_config)):
        for j in range(len(grid_config[0])):
            v_values[(i, j)] = 0
    for i in range(308):  # Iterations
        v_values, optimal_policy = modified_v_value_computer(grid_config, v_values)

    return optimal_policy
    # return sorted(optimal_policy.items())


def create_path(policy, start):
    path = [start]
    move = {0: lambda x: (x[0]+1, x[1]),
            1: lambda x: (x[0], x[1]+1),
            2: lambda x: (x[0]-1, x[1]),
            3: lambda x: (x[0], x[1]-1)}
    while policy[path[-1]] != 4:
        new_state = move[policy[path[-1]]](path[-1])
        if new_state in path:
            break
        path.append(new_state)
    return path


# print compute_v_values([[2, 2, 2, 2, 2, 2, 2, 2],
#                         [2, 0, 0, 1, 1, 0, 1, 2],
#                         [2, 0, 1, 0, 0, 0, 1, 2],
#                         [2, 0, 1, 0, 0, 0, 0, 2],
#                         [2, 0, 0, 0, 0, 0, 0, 2],
#                         [2, 0, 0, 0, 0, 0, 0, 2],
#                         [2, 1, 0, 0, 0, 0, 3, 2],
#                         [2, 2, 2, 2, 2, 2, 2, 2]])
print create_path(get_optimal_policy([[2, 2, 2, 2, 2, 2, 2, 2],
                                      [2, 0, 0, 1, 1, 0, 1, 2],
                                      [2, 0, 1, 0, 0, 0, 1, 2],
                                      [2, 0, 1, 0, 0, 0, 0, 2],
                                      [2, 0, 0, 0, 0, 0, 0, 2],
                                      [2, 0, 0, 0, 0, 0, 0, 2],
                                      [2, 1, 0, 0, 0, 0, 3, 2],
                                      [2, 2, 2, 2, 2, 2, 2, 2]]), (3, 5))

# print get_optimal_policy([[2, 2, 2, 2, 2, 2, 2, 2],
#                          [2, 0, 0, 1, 1, 0, 1, 2],
#                          [2, 0, 1, 0, 0, 0, 1, 2],
#                          [2, 0, 1, 0, 0, 0, 0, 2],
#                          [2, 0, 0, 0, 0, 0, 0, 2],
#                          [2, 0, 0, 0, 0, 0, 0, 2],
#                          [2, 1, 0, 0, 0, 0, 3, 2],
#                          [2, 2, 2, 2, 2, 2, 2, 2]])