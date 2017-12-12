import copy, time
class CSP_Solver(object):
    # CSP Solver using backtracking search
    # See assignment description for details regarding the return value of each method
    def __init__(self, evil_dict):
        self.evil_dict = evil_dict  # {student: [his archfiends!]}
        self.students = sorted(self.evil_dict)
        self.initial_domain = self.init_domain()
        self.init_arrangement = [["", "", "", ""],
                                 ["", "", "", ""],
                                 ["", "", "", ""],
                                 ["", "", "", ""]]
        self.crnt_arrangement = self.init_arrangement
        self.neighbours = {}  # mapping each seat to its neighbours {seat: [ neighbours ] }
        self.populate_neighbours()

    def valid_assignment(self, x, y, arrangement):
        student = arrangement[x][y]
        for nx, ny in self.neighbours[(x, y)]:  # adjacent seat (neighbour x, neighbour y)
            neighbour = arrangement[nx][ny]
            if neighbour in self.evil_dict[student]: return False  # There is an adjacent conflicting seat
        return True

    def init_domain(self):
        domain = {}
        for x in range(4):
            for y in range(4):
                domain[(x, y)] = self.students
        return domain

    @staticmethod
    def next_seat(x, y):
        if y == 3:  # seat is at the right most
            return x+1, 0  # next row, first column
        # elif x==3 and y==3: return 0, 0   # last seat
        else:
            return x, y+1  # next column, same row

    @staticmethod
    def backtrack(x, y):
        if y != 0:  # seat is at the left most
            return x, y - 1  # previous column, same row
        # elif x==0 and y==0: return 0, 0   # first seat (top left-corner seat)
        else:
            return x - 1, 3  # previous row, last column

    @staticmethod
    def get_adjacents(x, y):
        dxs = [-1, 0, 1]  # adjacent seats in a column
        dys = [-1, 0, 1]  # adjacent seats in a row
        for axis, daxis in [(x, dxs), (y, dys)]:  # checking if the student is at a border
            if axis == 0:
                daxis.pop(0)
            elif axis == 3:
                daxis.pop(-1)
        return dxs, dys

    def populate_neighbours(self):
        for x in range(4):
            for y in range(4):
                dxs, dys = self.get_adjacents(x, y)
                for dx in dxs:
                    for dy in dys:
                        if dx == 0 and dy == 0: continue
                        self.neighbours.setdefault((x, y), [])
                        self.neighbours[(x, y)].append((x + dx, y + dy))

    @staticmethod
    def assigned_students(arrangement):
        students = []
        for row in arrangement:
            for column in row:
                if column == "":
                    return sorted(students)
                else:
                    students.append(column)
        return sorted(students)

    def backtracking_search(self, arrangement):
        original_arrangement = copy.deepcopy(self.init_arrangement)
        x, y = 0, 0  # top left corner = first seat
        tried_students = {}  # {(x, y) : [students who have been assigned to this seat before
        # (while keeping track of parent assignments ===> line: 92)]}
        assignments = {}  # {student: 1 if assigned or 0 if not}
        for student in self.students: assignments[student] = 0
        while x <= 3 and y <= 3:
            valid_found = 0
            tried_students.setdefault((x, y), [])
            for student in self.students:
                if student not in tried_students[(x, y)] and assignments[student] == 0:
                    arrangement[x][y] = student
                    tried_students[(x, y)].append(student)
                else: continue
                if self.valid_assignment(x, y, arrangement):
                    valid_found = 1
                    std = arrangement[x][y]
                    assignments[std] = 1
                    break
            if valid_found == 0:
                std = arrangement[x][y]
                assignments[std] = 0
                arrangement[x][y] = ""
                tried_students[(x, y)] = []  # clearing previous assignments of that seat since its parent seat changed
                x, y = self.backtrack(x, y)
                std = arrangement[x][y]
                assignments[std] = 0
                arrangement[x][y] = ""
            else:
                x, y = self.next_seat(x, y)
        self.init_arrangement = original_arrangement
        return arrangement

    @staticmethod
    def get_backtracked_seat(seats_domain, assignments):
        seats_pool = sorted(seats_domain, key=lambda x: len(seats_domain[x]))  # seats sorted by length of domain
        x, y = seats_pool[0]
        for seat in seats_pool:
            if seat not in assignments:
                x, y = seat
                break
        return x, y

    def free_domains(self, assigning_function):
        temp_arrangement = copy.deepcopy(self.crnt_arrangement)
        domain_dict = copy.deepcopy(self.initial_domain)
        for row in range(4):
            for column in range(4):
                if temp_arrangement[row][column] != '':
                    domain_dict = assigning_function(((row, column), temp_arrangement[row][column]), domain_dict)
        return domain_dict

    def forward_checking(self, assignment, domain_dict):
        # self.students = sorted(self.evil_dict)[1:]  # To ignore the '' at the beginning
        x, y = assignment[0]
        student = assignment[1]
        self.delete_from_domains(x, y, student, domain_dict)
        for neighbour in self.neighbours[(x, y)]:
            adjacent_domain = domain_dict[neighbour]
            domain_dict[neighbour] = [mate for mate in adjacent_domain if
                                      mate not in self.evil_dict[student]]
        domain_dict[(x, y)] = [student]
        return domain_dict

    @staticmethod
    def valid_domain(domain, arcs):
        for seat in domain:
            if domain[seat] == []:
                return False
        return True

    def improved_backtracking(self, no_solution_check):
        arrangement = copy.deepcopy(self.init_arrangement)
        tried_students = {}  # {(x, y) : [students who have been assigned to this seat before]}
        assignments = []  # keeping track of parent assignments  e.g [(0,1), (1,2), ...]
        seats_domain = copy.deepcopy(self.initial_domain)
        backtracked = False
        arcs = []
        o = 0
        while 1:
            valid_found = 0  # Reference to finding at least one valid assignment, i.e that won't cause conflicts
            # if self.valid_domain(seats_domain):
            if not backtracked:
                x, y = self.get_backtracked_seat(seats_domain, assignments)
            if no_solution_check(seats_domain, arcs):  # detecting any empty domain for at least one seat
                tried_students.setdefault((x, y), [])
                for student in seats_domain[(x, y)]:
                    if student not in tried_students[(x, y)]:
                        arrangement[x][y] = student
                        tried_students[(x, y)].append(student)
                        seats_domain = self.forward_checking(((x, y), student), seats_domain)  # updating domain dict
                    else:
                        continue
                    assert self.valid_assignment(x, y, arrangement)
                    if no_solution_check == self.ac3:
                        arcs = []
                        xh, yh = self.get_backtracked_seat(seats_domain, assignments)
                        for neighbor in self.neighbours[(xh, yh)]:
                            arcs.append((neighbor, (xh, yh)))
                    assignments.append((x, y))
                    valid_found = 1
                    backtracked = False
                    break
            else:
                o += 1
            if valid_found == 0:
                assert arrangement[x][y] == ""
                assert (x, y) not in assignments
                backtracked = True
                # clearing previous assignments of that seat since its parent seat is going to change
                tried_students[(x, y)] = []
                x, y = assignments[-1]
                arrangement[x][y] = ""
                self.crnt_arrangement = arrangement
                seats_domain = self.free_domains(self.forward_checking)
                assignments.pop(-1)
            else:
                if len(assignments) == 16:
                    break
        print "Detected no solution %d times." %o
        return arrangement

    @staticmethod
    def delete_from_domains(x, y, student, domain):
        for dx in range(4):
            for dy in range(4):
                if dx == x and dy == y: continue
                try:
                    domain[(dx, dy)].remove(student)
                except ValueError:
                    pass

    def backtracking_with_forward_checking(self):
        return self.improved_backtracking(self.valid_domain)

    def init_arcs(self):
        """
        arcs :return: dictionary of each seat t as key and its h neighbours, where h has not been visited
          before by the arc t-->--h
        """
        arcs = []
        for x in range(4):
            for y in range(4):
                for neighbour in self.neighbours[(x, y)]:
                    arcs.append(( (x, y), neighbour ))
                    # print "(%d, %d)--->---(%d, %d)" %(x, y, neighbour[0], neighbour[1])
        return arcs

    def remove_inconsistent_values(self, seat_t, seat_h, domain):
        """
        :param seat_t: tail of the arc
        :param seat_h: head of the arc
        :param domain: dictionary
        :param assigned_students: dictionary of assigned students {student: 1 if assigned, 0 if not}
        :return: False iff for every value of X in domain(X) there is some allowed y in domain(Y)
        True otherwise, and removing every value of X in domain(X) that has no allowed y in domain(Y),
        :return: updated domain of seat_h
        """
        removed = False
        to_remove = []
        for x in domain[seat_t]:
            inconsistency = True
            for y in domain[seat_h]:
                # head and tail refers to the ends of the arc
                if not(y == x or y in self.evil_dict[x]):  # found a value y in domain(head) for x in domain(tail)
                    inconsistency = False
                    break
            if inconsistency:
                to_remove.append(x)
                removed = True
        for x in to_remove:
            domain[seat_t].remove(x)
        return removed

    def ac3(self, seats_domain, arcs):
        # arcs = []
        # seat, student = assignment
        # for neighbour in self.neighbours[seat]:
        #     arcs.append((neighbour, seat))
        # self.delete_from_domains(seat[0], seat[1], student, seats_domain)
        # seats_domain[seat] = [student]
        # arcs = self.init_arcs()
        while len(arcs) > 0:
            tail, head = arcs.pop(0)
            if self.remove_inconsistent_values(tail, head, seats_domain):
                for neighbour in self.neighbours[tail]:
                    arcs.append((neighbour, tail))
        return self.valid_domain(seats_domain, [])

    @staticmethod
    def update_arcs(seat, arcs):
        for arc in range(len(arcs)):
            if arcs[arc] == seat:
                arcs.pop(arc)
        return arcs

    def backtracking_with_ac3(self):
        return self.improved_backtracking(self.ac3)

def read_data():
    student_dict = {}  # {student: [his archfiends!]}
    with open('conflicts.txt', 'r') as f:
        for line in f:
            student, enemies = line.split('-')
            student_dict[student] = [enemy.strip() for enemy in enemies.split(',')]
    return student_dict

solver = CSP_Solver(read_data())

start_time = time.time()
print solver.backtracking_search(solver.init_arrangement)
print("backtracking_search:\t %s seconds\n" % (time.time() - start_time))

start_time = time.time()
print solver.backtracking_with_forward_checking()
print("backtracking_with_forward_checking:\t %s seconds\n" % (time.time() - start_time))

start_time = time.time()
print solver.backtracking_with_ac3()
print("backtracking_with_ac3:\t %s seconds\n" % (time.time() - start_time))

# ________________!USELESS DRAFT!_______________________, it wasn't useless for me though :P
# solution_domain = {(0,0):"Jack", (0,1):"Dan",(0,2):"Jill",(0,3):"Jeff",(1,0):"Joe",(1,1):"Dave",(1,2):"Bill"
#                                             ,(1,3):"Mike",(2,0):"Kim",(2,1):"Sam",(2,2):"Alan",(2,3):"Will",
#                                             (3,0):"Sue",(3,1):"Tome",(3,2):"Nick",(3,3):"John"}
# init_domain = solver.initial_domain
# init_domain[(0, 0)] = ["Alan"]
# assigned_students = {}
# for student in solver.evil_dict.keys():
#     assigned_students[student] = 0
# assigned_students["Alan"] = (0,0)
# print solver.remove_inconsistent_values((0,0), (0,1), init_domain, assigned_students)
# arrangement = solver.init_arrangement
# for s in solution_domain:
#     x, y = s
#     arrangement[x][y] = solution_domain[s]
#     solution_domain[s] = [solution_domain[s]]
# arrangement[0][2] = ""
# solver.crnt_arrangement = arrangement
# solution_domain = solver.forward_checking(((0, 2), ""), solution_domain)
# arrangement[1][2] = ""
# solver.crnt_arrangement = arrangement
# solution_domain = solver.forward_checking(((1, 2), ""), solution_domain)

# print solver.forward_checking(((0, 0), ""), solution_domain)
# Solution
# print solver.get_total_conflicts(
# [['Jack', 'Dan', 'Jill', 'Jeff'],
# ['Joe', 'Dave', 'Bill', 'Mike'],
# ['Kim', 'Sam', 'Alan', 'Will'],
# ['Sue', 'Tom', 'Nick', 'John']])
