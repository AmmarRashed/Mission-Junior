# Ammar Rashed
# 214200715

"""QUESTION 1"""
def get_route(lineage, goal, depth):
    route = [goal]
    for i in range(depth-1):
        goal = lineage[goal]
        route.append(goal)
    return route[::-1]


def bestRoute(mountain):
    # init matrix  O( ((k+1)*(k+2)/2) - 1 )
    # In java we would just init a 2D array int[][] m = new int[k][k], but it will take almost double the size.
    lineage = {}  # A dictionary of k keys holding the Route from the Top of the mountain to the bottom
    m = [[mountain[0][0], float('-inf')]]
    for i in range(1, len(mountain)):
        lineage[(i, 0)] = (i-1, 0)
        m.append([mountain[i][0] + m[i-1][0]])
        for j in range(len(mountain[i])):
            m[i].append(float('-inf'))
    goal = (len(mountain)-1, 0)
    res = m[len(mountain)-1][0]
    for i in range(1, len(m)):
        for j in range(1, len(mountain[i])):
            if m[i-1][j-1] > m[i-1][j]:
                new = m[i-1][j-1]
                lineage[(i, j)] = (i-1, j-1)
            else:
                new = m[i - 1][j]
                lineage[(i, j)] = (i - 1, j)
            new += mountain[i][j]
            if i == len(m)-1 and new > res:  # Is this the bottom of the mountain yet?
                res = new
                goal = (i, j)
            m[i][j] = new
    route = get_route(lineage, goal, len(mountain))
    print len(lineage)
    # complexity is len(lineage) which is [(k(k-1)/2) - 1]  +  [((k+1)*(k+2)/2) - 1] (for initialization)
    # O(k**2)
    return route, res


Mountain = [[1], [1, 2], [3, 5, -4], [-10, -6, -7, 1]]
Route, Score = bestRoute(Mountain)
print "Route: ", Route
print "Score: ", Score
# ____________________________________________________________________________________

"""QUESTION 2"""
def not_conflicting(conferences, c1, c2):
    start1, end1 = conferences[c1][0],conferences[c1][1]
    start2, end2 = conferences[c2][0],conferences[c2][1]
    if (start1 <= start2 < end1) or (start1 < end2 <= end1):
        return 0
    return 1

def bestSelection(conferences):
    ary = []
    # Sorting conferences by their ending time
    conferences_sorted = sorted(conferences.items(), key=lambda (key, value): value[1])
    # Initializing the array, with the number of participants in each conference
    for con in conferences_sorted:
        ary.append(con[1][2])
    combination = [[conferences_sorted[0][0]]]
    for i in range(1, len(ary)):
        c1 = conferences_sorted[i][0]
        combination.append([c1])
        for j in range(i):
            c2 = conferences_sorted[j][0]
            if not_conflicting(conferences, c1, c2):
                res = conferences_sorted[i][1][2] + ary[j]
                if ary[i] < res:
                    combination[i] = combination[j] + [c1]
                    ary[i] = res
    max_indx, max_value = max(enumerate(ary), key=lambda v:v[1])
    return combination[max_indx], max_value

conferences = {"Conference 1":	[1300,	1559,	300],
               "Conference 2":	[1100, 1359,	500],
               "Conference 3":	[1600,	1759,	200]}

conferences2 = {"Conference 1": [1000, 1159, 200],
                "Conference 2": [1100, 1259, 400],
                "Conference 3": [1200, 1359, 300],
                "Conference 4": [1000, 1459, 500]}

cs, n = bestSelection(conferences2)
print "Selected conferences:\t" + str(cs)
print "Total number of participants:\t" + str(n)
# ____________________________________________________________________________________

"""QUESTION 3"""
def print_matrix(m):
    for i in m:
        for j in i:
            print j.combination,
        print


class Comb:
    def __init__(self, value, combination):
        self.value = value
        self.combination = combination


def possibleCombinations(n):
    if n < 2:
        return 0
    m = []
    # Initializing the matrix
    for i in range(n-1):
        m.append([Comb(value=1, combination=[' '])])
        m[0].append(Comb(value=1, combination=['1+'*(i+1)]))
    m[0].append(Comb(value=1, combination=['1+'*n]))
    # print_matrix(m)
    # Finding the solution
    for i in range(1, n-1):
        for j in range(1, n+1):
            m[i].append(None)
            if j >= i+1:
                value = m[i-1][j].value + m[i][j-(i+1)].value
                combination = m[i - 1][j].combination
                new_comb = str(i+1)+"+"
                for c in m[i][j-(i+1)].combination:
                    combination.append(new_comb + c)
                m[i][j] = Comb(value, combination)
                # print_matrix(m)
            else:
                m[i][j] = m[i-1][j]
    # print print_combinations(m)
    assert len(m) == n - 1
    assert len(m[0]) - 1 == n
    result = m[len(m)-1][len(m[0])-1]
    return result.value, [result.combination[i].strip()[:-1] for i in range(len(result.combination)-1, -1, -1)]


num, ways = possibleCombinations(4)
print "There are %d combinations" % num
for way in ways:
    print way
