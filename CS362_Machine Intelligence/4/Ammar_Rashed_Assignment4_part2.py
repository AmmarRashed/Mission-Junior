reward = {0: 0, 1: -5, 2: -10, 3: 30}
noises = [0.7, 0.15, 0.15]


def east(state, grid_config, with_noise=0):
    x, y = state
    # Going east
    if y == len(grid_config[0]) - 1:  # east edge
        result = state
    else:
        result = x, y+1
    if with_noise:
        return result, north(state, grid_config, 0), south(state, grid_config, 0)
    return result


def west(state, grid_config, with_noise=0):
    x, y = state
    # Going west
    if y == 0:  # west edge
        result = state
    else:
        result = x, y - 1
    if with_noise:
        return result, north(state, grid_config, 0), south(state, grid_config, 0)
    return result


def north(state, grid_config, with_noise=0):
    x, y = state
    # Going north
    if x == len(grid_config) - 1:  # Top edge
        result = state
    else:
        result = x + 1, y
    if with_noise:
        return result, east(state, grid_config, 0), west(state, grid_config, 0)
    return result


def south(state, grid_config, with_noise):
    x, y = state
    # Going south
    if x == 0:  # Bottom edge
        result = state
    else:
        result = x-1, y
    if with_noise:
        return result, east(state, grid_config, 0), west(state, grid_config, 0)
    return result


def get_reward(agent_state, adv_state, grid_config):
    global reward
    if agent_state == adv_state:
        return -50
    x, y = agent_state
    return reward[grid_config[-x-1][y]]


agent_policy = {}
adv_policy = {}


def construct_mm_tree(agent_state, adv_state, grid_config, depth_limit, noise, max_min=True):
    global noises
    # global agent_policy
    # global adv_policy
    if max_min:  # maximizer
        state = agent_state
        maximizer = float('-inf')
    else:  # minimizer
        state = adv_state
        minimizer = float('inf')

    state_reward = get_reward(agent_state, adv_state, grid_config) * noise
    if depth_limit == 0 or state_reward != 0:  # base case or terminal state
        return state_reward

    i = 0
    for action in get_successor_actions(state, grid_config, with_noise=max_min):  # north, east, south and west
        if max_min:  # maximizer
            child_reward = 0
            assert len(action) == 3  # ( (x1,y1), (x2,y2), (x3,y3) )
            for st in range(len(action)):
                child_reward += construct_mm_tree(action[st], adv_state, grid_config, depth_limit-1, noises[st], False)\
                                * noises[st]
            if child_reward > maximizer:
                maximizer = child_reward
                # agent_policy[state] = i
        else:
            assert len(action) == 2  # (x, y) of one state, because there is no stochasticity
            child_reward = construct_mm_tree(agent_state, action, grid_config, depth_limit - 1, noise=1)
            if child_reward < minimizer:
                minimizer = child_reward
                # adv_policy[state] = i
        i += 1
    if max_min:  # maximizer
        return maximizer
    return minimizer


def construct_mm_tree_alpha_beta_pruning(agent_state, adv_state, grid_config, depth_limit,
                                         comparator, alpha, beta, noise, max_min=True):
    global noises
    # global agent_policy
    # global adv_policy
    if max_min:  # maximizer
        state = agent_state
        maximizer = float('-inf')
    else:  # minimizer
        state = adv_state
        minimizer = float('inf')

    state_reward = get_reward(agent_state, adv_state, grid_config) * noise
    if depth_limit == 0 or state_reward != 0:  # base case or terminal state
        return state_reward

    i = 0
    for action in get_successor_actions(state, grid_config, with_noise=max_min):  # north, east, south and west
        if max_min:  # maximizer
            child_reward = 0
            assert len(action) == 3  # ( (x1,y1), (x2,y2), (x3,y3) )
            for st in range(len(action)):
                child_reward += construct_mm_tree_alpha_beta_pruning(action[st], adv_state, grid_config, depth_limit-1,
                                  comparator=maximizer, alpha=alpha,beta=beta, noise=noises[st], max_min=False)\
                                * noises[st]
            if child_reward > maximizer:
                maximizer = child_reward
                # agent_policy[state] = i
            if child_reward > alpha:
                alpha = child_reward
            if comparator is not None and child_reward > comparator:
                break
        else:
            assert len(action) == 2  # (x, y) of one state, because there is no stochasticity
            child_reward = construct_mm_tree_alpha_beta_pruning(agent_state, action, grid_config, depth_limit - 1,
                                                                comparator=minimizer, alpha=alpha, beta=beta, noise=1)
            if child_reward < minimizer:
                minimizer = child_reward
                # adv_policy[state] = i
            if child_reward < beta:
                beta = child_reward
            if child_reward < comparator:
                break
        i += 1
    if max_min:  # maximizer
        return maximizer
    return minimizer


def get_successor_actions(state, grid_config, with_noise=True):
    return north(state, grid_config,with_noise),\
           east(state, grid_config, with_noise),\
           south(state, grid_config,with_noise),\
           west(state, grid_config, with_noise)


def manhattan_distance(agent_state, other_state):  # other_state can be either goal state or adversary
    xa, ya = agent_state
    xv, yv = other_state
    return abs(xa-xv) + abs(ya-yv)


def emm_no_pruning(grid_config, depth_limit):
    start_agent = (6, 1)
    start_adv = (1, 1)
    return construct_mm_tree(start_agent, start_adv, grid_config, depth_limit*2, 1)

def emm_ab_pruning(grid_config, depth_limit):
    start_agent = (6, 1)
    start_adv = (1, 1)
    return construct_mm_tree_alpha_beta_pruning(start_agent, start_adv, grid_config, depth_limit * 2,comparator=None,
                                                alpha=float('-inf'),beta=float('inf'), noise=1)


def compute_max_q_values(state, adv_state, grid_config, w1, w2):
    reward = get_reward(state, adv_state, grid_config)
    if reward != 0:  # terminal state
        return w1*manhattan_distance(state, (1, 6))+ w2*manhattan_distance(state, adv_state)
    global noises
    max_q = float('-inf')
    succeessor_states = get_successor_actions(state, grid_config, True)
    for s_prime in succeessor_states:
        f1 = 0
        f2 = 0
        for s in range(len(s_prime)):
            # if grid_config[s_prime[s][0]][s_prime[s][1]] != 0:
            #     continue
            f1 += manhattan_distance(s_prime[s], (1, 6)) * noises[s]
            f2 += manhattan_distance(s_prime[s], adv_state) * noises[s]
        q = w1*f1 + w2*f2
        if q > max_q:
            max_q = q
    return max_q


def get_adv_state(agent_state, adv_start_state, grid_config):
    best_adv_action = None
    min_man_distance = float('inf')
    for state in get_successor_actions(adv_start_state, grid_config, with_noise=False):
        dist = manhattan_distance(agent_state, state)
        if min_man_distance > dist:
            min_man_distance = dist
            best_adv_action = state
    return best_adv_action


def approximate_q_learning(grid_config, gamma=1, alpha=0.1, start_adv=(1, 1)):
    global reward, noises
    q_values = {}
    ws = [1, 1]
    # for x in range(len(grid_config)):
    #     for y in range(len(grid_config[0])):
    x, y = 6, 1
    # if grid_config[x][y] != 0: continue
    successor_states = get_successor_actions((x, y), grid_config, with_noise=True)
    for action in range(len(successor_states)):  # north, east, south and west
        # assert grid_config[x][y] == 0
        f1 = 0
        f2 = 0
        for state in range(len(successor_states[action])):
            adv_state = get_adv_state(successor_states[action][state], start_adv, grid_config)
            f1 += manhattan_distance(successor_states[action][state], (1, 6)) * noises[state]
            f2 += manhattan_distance(successor_states[action][state], adv_state) * noises[state]
        fs = (f1, f2)
        q_values[((x, y), action)] = 0
        for i in range(len(fs)):
            q_values[((x, y), action)] += ws[i]*fs[i]
        target = 0
        for s in range(len(successor_states[action])):
            adv_state = get_adv_state(successor_states[action][s], start_adv, grid_config)
            r = get_reward(successor_states[action][s], adv_state, grid_config)
            target += (r + gamma * compute_max_q_values(successor_states[action][s], adv_state,
                                                        grid_config, ws[0], ws[1])) * noises[s]
        difference = target - q_values[((x, y), action)]
        q_values[((x, y), action)] += alpha*difference
        for i in range(len(fs)):
            ws[i] += alpha*difference*fs[i]
    return ws[0], ws[1]


grid_config_input = [[2, 3, 2, 2, 2, 2, 2, 2],
                    [2, 0, 0, 0, 1, 0, 1, 2],
                    [2, 2, 3, 0, 2, 0, 1, 2],
                    [2, 0, 0, 2, 0, 0, 0, 2],
                    [2, 0, 0, 0, 0, 0, 0, 2],
                    [2, 0, 0, 0, 0, 0, 0, 2],
                    [2, 1, 0, 0, 0, 0, 0, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2]]
import time
start = time.time()
print emm_no_pruning(grid_config_input, 2)
print "It took: \t",
print time.time() - start
start = time.time()
print emm_ab_pruning(grid_config_input, 2)
print "It took: \t",
print time.time() - start
print approximate_q_learning(grid_config_input, start_adv=(1,1))
