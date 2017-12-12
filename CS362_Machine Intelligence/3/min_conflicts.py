import random,copy, time
random.seed(1)


class CSP_Solver(object):
    # CSP Solver using min conflicts algorithm
    # See assignment description for details regarding the return value of each method
    def __init__(self, evil_dict):
        self.evil_dict = evil_dict  # {student: [his archfiends!]}
        self.conflicting_seats = {}  # { student : 1 }, a dictionary prevents duplicates keys
        self.lst_arrangement = [["Alan", "Bill", "Jack", "Jeff"],
                                ["Dan", "Dave", "Jill", "Joe"],
                                ["John", "Kim", "Sam", "Sue"],
                                ["Mike", "Nick", "Tom", "Will"]]

    @staticmethod
    def get_student_index(student, arrangement):
        for row in range(len(arrangement)):
            try:
                y = arrangement[row].index(student)
                x = row
            except ValueError:
                continue
            return x, y

    @staticmethod
    def get_adjacents(x, y):
        dxs = [-1, 0, 1]  # moves a student can make in a column
        dys = [-1, 0, 1]  # moves a student can make in a row
        for axis, daxis in [(x, dxs), (y, dys)]:  # checking if the student is at a border
            if axis == 0:
                daxis.pop(0)
            elif axis == 3:
                daxis.pop(-1)
        return dxs, dys

    def get_num_of_conflicts(self, student, arrangement):
        x, y = self.get_student_index(student, arrangement)
        conflicts = 0
        # print "selected student is: ", student
        # print "--------"
        dxs, dys = self.get_adjacents(x, y)
        for dx in dxs:
            for dy in dys:
                if dx == 0 and dy == 0: continue
                dude = arrangement[x+dx][y+dy]
                if dude in self.evil_dict[student]:
                    conflicts += 1
        if conflicts > 0: self.conflicting_seats[student] = (x, y)
        return conflicts

    def get_total_conflicts(self, arrangement):
        self.conflicting_seats = {}
        conflicts = 0
        for row in range(4):
            for column in range(4):
                conflicts += self.get_num_of_conflicts(arrangement[row][column], arrangement)
        return conflicts

    @staticmethod
    def find_a_random_student(arrangement):
        return arrangement[random.randint(0, 3)][random.randint(0, 3)]

    @staticmethod
    def swap_seats(arrangement, x, y, dx, dy):
        temp_arrangement = copy.deepcopy(arrangement)
        temp_seat = temp_arrangement[x][y]
        temp_arrangement[x][y] = temp_arrangement[dx][dy]
        temp_arrangement[dx][dy] = temp_seat
        return temp_arrangement

    def get_best_arrangement(self, student, current_arrangement):
        x, y = self.get_student_index(student, current_arrangement)
        min_conflicts = self.get_total_conflicts(current_arrangement)
        best_arrangement = copy.deepcopy(current_arrangement)
        for dx in range(4):  # That will give the priority to the min row (in case of a tie)
            for dy in range(4):  # That will give the priority to the min column (in case of a tie)
                proposed_arrangement = self.swap_seats(current_arrangement, x, y, dx, dy)
                if dx == x and dy == y:
                    continue
                proposed_conflicts = self.get_total_conflicts(proposed_arrangement)
                if proposed_conflicts <= min_conflicts:
                    min_conflicts = proposed_conflicts
                    best_arrangement = proposed_arrangement
        return best_arrangement, min_conflicts

    @staticmethod
    def min_row_column(seats, row_column):
        return min(seats, key=lambda x: seats[x][row_column])

    def solve_csp(self, arrangement):
        counter = 0
        conflicts = self.get_total_conflicts(arrangement)
        conflicting_seats = self.conflicting_seats.keys()
        while counter < 1000:  # If the algorithm converges to an incomplete solution, return it
            if conflicts == 0:  # A valid arrangement
                return arrangement
            picked_student = conflicting_seats[random.randint(0, len(conflicting_seats)-1)]
            # temp_conflicts = conflicts
            arrangement, conflicts = self.get_best_arrangement(picked_student, arrangement)
            # if temp_conflicts == conflicts: # if there is a tie, pick min row index
            #     picked_student = self.min_row_column(self.conflicting_seats, row_column=0)
            #     temp_conflicts = conflicts
            #     arrangement, conflicts = self.get_best_arrangement(picked_student, arrangement)
            #     if temp_conflicts == conflicts:  # if there is a tie, pick min column index
            #         picked_student = self.min_row_column(self.conflicting_seats, row_column=1)
            #         arrangement, conflicts = self.get_best_arrangement(picked_student, arrangement)
            counter += 1
        print "Best found arrangement has %d conflicts" % conflicts
        return arrangement


def read_data():
    student_dict = {}  # {student: [his archfiends!]}
    with open('conflicts.txt', 'r') as f:
        for line in f:
            student, dudes = line.split('-')
            student_dict[student] = [dude.strip() for dude in dudes.split(',')]
    return student_dict

solver = CSP_Solver(read_data())
start_time = time.time()
print solver.solve_csp(solver.lst_arrangement)
print("min conflicts:\t %s seconds" % (time.time() - start_time))

# Various solutions found while debugging!
# print solver.get_total_conflicts([['Jeff', 'Jill', 'Dan', 'Jack'], ['Mike', 'Bill', 'Dave', 'Joe'], ['Will', 'Alan', 'Sam', 'Kim'], ['John', 'Nick', 'Tom', 'Sue']])
# print solver.get_total_conflicts([['Jack', 'Dan', 'Jill', 'Jeff'], ['Joe', 'Dave', 'Bill', 'Mike'], ['Kim', 'Sam', 'Alan', 'Will'], ['Sue', 'Tom', 'Nick', 'John']])
# print solver.get_total_conflicts([['Sue', 'Kim', 'Joe', 'Jack'], ['Tom', 'Sam', 'Dave', 'Dan'], ['Nick', 'Alan', 'Bill', 'Jill'], ['John', 'Will', 'Mike', 'Jeff']])
# print solver.get_total_conflicts([['Jack', 'Joe', 'Kim', 'Sue'], ['Dan', 'Dave', 'Sam', 'Tom'], ['Jill', 'Bill', 'Alan', 'Nick'], ['Jeff', 'Mike', 'Will', 'John']])