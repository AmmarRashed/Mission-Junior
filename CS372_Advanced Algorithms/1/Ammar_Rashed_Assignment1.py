import time
import matplotlib.pyplot as plt
# ___ QUESTION 1 ______
def bag_switch(outcomes):
    bags = {1: ['B', 'Y', 'G'],
            2: ['W', 'R', 'O']}
    try:
        if outcomes[0] in bags[2]: return 0
    except IndexError:  # outcomes sequence is empty
        return 0

    if outcomes[1] in bags[2]:
        return 1
    i = 1
    bag1_pointer = 0
    o = 0
    while 1:
        try:
            o += 1
            if outcomes[i] in bags[1]:
                bag1_pointer = i
                if outcomes[i+1] in bags[2]:
                    print "There are %d iterations when N*2 is %d" % (o, (i+1)*2)
                    return i+1
                else:
                    i *= 2
            else:  # outcomes[i] in bags[2]
                if outcomes[i-1] in bags[1]:
                    print "There are %d iterations when N*2 is %d" % (o, i * 2)
                    return i
                i -= (i-bag1_pointer)/2
        except IndexError:
            return None
# ___ Question 1 Analysis ________
# Worst case case => O(2*log(n)) where n = 2 * N
# y = []
# x = []
# i = 1
# while i<20:
#     i += 1
#     outcomes = 'Y '*((2**i)+2)+'W '*((2**i)+2)
#     x.append((2 ** i) + 2)
#     input = outcomes.split(' ')[:-1]
#     start = time.time()
#     bag_switch(input)
#     y.append(time.time() - start)#
# print y,'\n'
# plt.plot(x, y, label="Worst Case")
#
# Best case Complexity => O(log(n)) where n = 2 * N
# y = []
# x = []
# i = 1
# while i<20:
#     i += 1
#     outcomes = 'Y '*(2**i)+'W '*(2**i)
#     x.append(2 ** i)
#     input = outcomes.split(' ')[:-1]
#     start = time.time()
#     bag_switch(input)
#     y.append(time.time() - start)
#
# print y
#
# plt.plot(x, y, label="Best Case")
# plt.legend(bbox_to_anchor=(0.8, 1.1), loc=2, borderaxespad=0)
# plt.show()

'''___________________________________________________________________________________________'''
# _____  QUESTION 2 ___________
def end_left(numbers):  # The end of the left subsequence
    for i in range(len(numbers)-1):
        if numbers[i+1] < numbers[i]:
            return i
    # assert i + 1 == len(numbers) - 1
    return i+1  # Array is already sorted

def start_right(numbers):  # The start of the right subsequence
    for i in range(len(numbers)-1, 0, -1):
        if numbers[i - 1] > numbers[i]:
            return i
    # assert i - 1 == 0
    return 0  # Array is already sorted

def stretch_left(numbers, min_index, start):
    temp = numbers[min_index]
    if start < 0:
        return 0
    if numbers[start-1] <= temp:
        return start
    return stretch_left(numbers, min_index, start-1)
    # for i in range(start-1, -1, -1):
    #     if numbers[i] <= temp:
    #         return i+1
    # return 0

def stretch_right(numbers, max_index, start):
    temp = numbers[max_index]
    if start > len(numbers)-1:
        return len(numbers)-1
    if numbers[start] >= temp:
        return start - 1
    return stretch_right(numbers, max_index, start+1)
    # for i in range(start, len(numbers)):
    #     if numbers[i] >= temp:
    #         return i - 1
    # return len(numbers) - 1

def partition_list(numbers):
    # Initializing the left subsequence
    left_end = end_left(numbers)
    if left_end >= len(numbers)-1:
        print "Array is already sorted"
        return None, None

    # Initializing the right subsequence
    right_start = start_right(numbers)
    # print "left end is %d\nright start is %d" % (left_end, right_start)
    # print numbers[left_end] , numbers[right_start]

    # Finding the max and the min elements in the middle subsequence
    min_index = left_end + 1
    max_index = right_start - 1
    for i in range(left_end, right_start+1):
        if numbers[i] < numbers[min_index]: min_index = i
        if numbers[i] > numbers[max_index]: max_index = i
    # print "max is %d\nmin is %d" %(max_index,min_index)

    # Stretch the middle subsequence from the left
    mid_start = stretch_left(numbers, min_index, left_end)
    # Stretch the middle subsequence from the right
    mid_end = stretch_right(numbers, max_index, right_start)

    return mid_start, mid_end

# numbers = [1, 2, 4, 7, 10, 11, 7, 12, 6, 7, 16, 18, 19]
numbers = [1,3,4,6,8,10,13,15,11,9,12,13,15,17,20,25]
# print end_left(numbers)
# print start_right(numbers)
x, y = partition_list(numbers)
print x, y

'''___________________________________________________________________________________________'''
#  ___________ QUESTION 3 _____________
class Graph:
    def __init__(self):
        self.graph_table = {}  # {node : [its children]}

    def add_edge(self, node_start, node_end):
        self.graph_table.setdefault(node_start, [])
        self.graph_table[node_start].append(node_end)


def create_graph(G):
    graph = Graph()
    for start, end in G:
        graph.add_edge(start, end)
    return graph.graph_table


def crawl(gt, start, end):
    global it
    it += 1
    paths = 0
    if start == end:
        return 1
    for e in gt[start]:
        paths += crawl(gt, e, end)
    return paths


def number_of_paths(G, start, end):
    gt = create_graph(G)
    if start == end:
        return 1
    return crawl(gt, start, end)

G = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('C', 'E'), ('D', 'E')]
start = 'A'
end = 'E'
print number_of_paths(G, start, end)