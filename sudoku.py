import numpy
import random
random.seed()

#  kích thước ma trận
NumDigits = 9


# tạo class Population, quần thể cho bài toán
class Population(object):

    # khởi tạo ứng cử viên
    def __init__(self):
        # thuộc tính candidates được khởi tạo thành 1 mảng rỗng
        self.candidates = []
        return

    # khởi tạo cá thể
    def seed(self, size, original_Comparison):
        self.candidates = []
        # Xác định các giá trị có thể có mà mỗi ô vuông có thể nhận
        helper = Candidate()
        helper.values = [[[] for j in range(0, NumDigits)]
                         for i in range(0, NumDigits)]
        for row in range(0, NumDigits):
            for column in range(0, NumDigits):
                for value in range(1, 10):
                    if((original_Comparison.values[row][column] == 0) and not (original_Comparison.isColumnDuplicate(column, value) or original_Comparison.isBlockDuplicate(row, column, value) or original_Comparison.isRowDuplicate(row, value))):
                        # Giá trị đã tồn tại.
                        helper.values[row][column].append(value)
                    elif(original_Comparison.values[row][column] != 0):
                        # Trùng giá trị của đề bài
                        helper.values[row][column].append(
                            original_Comparison.values[row][column])
                        break

        # Nhân giống một quần thể mới.
        for p in range(0, size):
            g = Candidate()
            for i in range(0, NumDigits):  # Hàng mới trong ứng cử viên.
                row = numpy.zeros(NumDigits, dtype=int)

                # điền vào givens.
                # Giá trị cột j mới trong hàng i.
                for j in range(0, NumDigits):

                    # Nếu giá trị đã được đưa ra, đừng thay đổi nó.
                    if(original_Comparison.values[i][j] != 0):
                        row[j] = original_Comparison.values[i][j]
                    # Điền vào khoảng trống bằng bảng trợ giúp.
                    elif(original_Comparison.values[i][j] == 0):
                        row[j] = helper.values[i][j][random.randint(
                            0, len(helper.values[i][j])-1)]

                # Nếu chúng tôi không có bảng hợp lệ, hãy thử lại. Không được có bản sao trong hàng.
                while(len(list(set(row))) != NumDigits):
                    for j in range(0, NumDigits):
                        if(original_Comparison.values[i][j] == 0):
                            row[j] = helper.values[i][j][random.randint(
                                0, len(helper.values[i][j])-1)]

                g.values[i] = row

            self.candidates.append(g)

        # Tính toán Fitness của tất cả các ứng cử viên trong dân số.
        self.updateFitness()
        print("Seeding complete.")
        return

    # Cập nhật Fitness của mọi ứng cử viên / nhiễm sắc thể
    def updateFitness(self):
        for candidate in self.candidates:
            candidate.updateFitness()
        return

    # Sắp xếp dân số dựa trên Fitness.
    def sort(self):
        for i in range(len(self.candidates)-1):
            max = i
            for j in range(i+1, len(self.candidates)):
                if self.candidates[max].fitness < self.candidates[j].fitness:
                    max = j
            temp = self.candidates[i]
            self.candidates[i] = self.candidates[max]
            self.candidates[max] = temp
        return


# tạo class Candidate, ứng cử viên cho bài toán
class Candidate(object):

    # khởi tạo
    def __init__(self):
        # khởi tạo values bằng 1 ma tran NumDigits x NumDigits có giá trị ban đâu là 0
        self.values = numpy.zeros((NumDigits, NumDigits), dtype=int)
        # khởi tạo fitness, số thích nghi bằng None
        self.fitness = None
        return

    # hàm tính fitness của đối tượng đang xét
    # sẽ xác định dựa trên các ràng buộc của bài toán
    def updateFitness(self):

        # khởi tạo mảng 9 phần tự để đếm số số giá trị trùng, giá trị ban đầu bằng 0
        columnCount = numpy.zeros(NumDigits, dtype=int)
        rowCount = numpy.zeros(NumDigits, dtype=int)
        blockCount = numpy.zeros(NumDigits, dtype=int)
        rowSum = 0
        columnSum = 0
        blockSum = 0

        # i chạy qua các cột có giá trị từ 0 - (NumDigits - 1)
        # để tính ra giá trị colmnSum
        # i là cột
        # j là hàng
        for i in range(0, NumDigits):

            # khởi tạo biến dò khác không
            # nếu = 0 thì biến sẽ tăng lên 1
            nonzero = 0

            # j chạy qua các hàng, có giá trị từ 0 - (NumDigits - 1)
            for j in range(0, NumDigits):
                # cập nhập lại ds với số xuất hiện cụ thể
                # sau khi chạy xong 1 cột
                # nếu trong trường hợp đúng là columnCount sẽ mang giá trị toàn là số 1
                # nếu có giá trị nào trùng vị ở vị trí đó trong columnCount tăng lên 1
                columnCount[self.values[j][i]-1] += 1

            #columnSum = columnSum + (1/len(set(columnCount)))/NumDigits

            # tính giá trị columnSum
            # for cho k chạy từ 0 -> (NumDigits - 1)
            # nếu chạy xong for, kq đúng thì nó sẽ trả về nonzero = NumDigits
            for k in range(0, NumDigits):

                # nếu mà trong for bên trên chạy columnCount có giá trị tương ứng ở vị trí k thì biến nonzero tăng lên 1
                if columnCount[k] != 0:
                    nonzero += 1
            # nếu đúng thì nonzero/NumDigits = 1
            nonzero = nonzero/NumDigits
            columnSum = (columnSum + nonzero)
            # khởi tão lại columnCount và chạy tiếp cột tiếp theo
            columnCount = numpy.zeros(NumDigits, dtype=int)

        # sau khi chạy hết for, sẽ thu được tối đa columnSum = 9 => columnSum/NumDigits = 1
        columnSum = columnSum/NumDigits

        # tương tự làm thế với hàng để tính ra rowSum
        for i in range(0, NumDigits):
            nonzero = 0
            for j in range(0, NumDigits):
                # chú ý vị trí của ma trận
                rowCount[self.values[i][j]-1] += 1

            #rowSum = rowSum + (1/len(set(rowCount)))/NumDigits
            for k in range(0, NumDigits):
                if rowCount[k] != 0:
                    nonzero += 1
            nonzero = nonzero/NumDigits
            rowSum = (rowSum + nonzero)
            rowCount = numpy.zeros(NumDigits, dtype=int)

        rowSum = rowSum/NumDigits

        # tiếp tục với khối 3x3 bên trong
        # for mỗi lần lặp, giá trị i sẽ tăng thêm 3
        for i in range(0, NumDigits, 3):
            # for mỗi lần lặp, giá trị j sẽ tăng thêm 3
            for j in range(0, NumDigits, 3):
                blockCount[self.values[i][j]-1] += 1
                blockCount[self.values[i][j+1]-1] += 1
                blockCount[self.values[i][j+2]-1] += 1

                blockCount[self.values[i+1][j]-1] += 1
                blockCount[self.values[i+1][j+1]-1] += 1
                blockCount[self.values[i+1][j+2]-1] += 1

                blockCount[self.values[i+2][j]-1] += 1
                blockCount[self.values[i+2][j+1]-1] += 1
                blockCount[self.values[i+2][j+2]-1] += 1

                #blockSum = blockSum + (1/len(set(blockCount)))/NumDigits
                nonzero = 0
                for k in range(0, NumDigits):
                    if blockCount[k] != 0:
                        nonzero += 1
                nonzero = nonzero/NumDigits
                blockSum = blockSum + nonzero
                blockCount = numpy.zeros(NumDigits, dtype=int)
        blockSum = blockSum/NumDigits

        # nếu cả 3 ràng buộc đều = 1 thì cá thế đó chính là cá thể cần tìm
        if (int(columnSum) == 1 and int(blockSum) == 1 and rowSum == 1):
            fitness = 1.0
        # ngược lại sẽ tính ra fitness để đánh giá đem đi lai ghép, ... tạo ra thế hệ kế tiếp
        else:
            fitness = (rowSum * blockSum * columnSum)

        self.fitness = fitness
        return

    # đột biến
    # trong phần này, ta sẽ chọn ra 1 hàng và hoán đổi 2 giá trị với nhau của hàng đó
    def mutate(self, mutationRate, original_Comparison):

        # random.uniform() cho ra số float
        r = random.uniform(0, 1.1)
        # lấy ra giá trị r > 1
        while(r > 1):
            r = random.uniform(0, 1.1)

        # khởi tạo biến thành công, mặc định đang là sai
        success = False
        # bắt đầu đi đột biến
        if (r < mutationRate):
            while(not success):
                # random.randint trả về số nguyên
                row = random.randint(0, 8)

                # tìm 2 cột khác nhau
                fromColumn = random.randint(0, 8)
                toColumn = random.randint(0, 8)
                while(fromColumn == toColumn):
                    fromColumn = random.randint(0, 8)
                    toColumn = random.randint(0, 8)

                # kiểm tra xem 2 vị trí có trống không (khác vị trí của đề bài)
                if(original_Comparison.values[row][fromColumn] == 0 and original_Comparison.values[row][toColumn] == 0):
                    # kiểm tra xem giá trị khi hoán đổi có làm cho bị trùng giá trị không
                    if(not original_Comparison.isColumnDuplicate(toColumn, self.values[row][fromColumn])
                       and not original_Comparison.isColumnDuplicate(fromColumn, self.values[row][toColumn])
                       and not original_Comparison.isBlockDuplicate(row, toColumn, self.values[row][fromColumn])
                       and not original_Comparison.isBlockDuplicate(row, fromColumn, self.values[row][toColumn])):

                        # hoán đổi 2 giá trị
                        temp = self.values[row][toColumn]
                        self.values[row][toColumn] = self.values[row][fromColumn]
                        self.values[row][fromColumn] = temp
                        success = True

        return success


# class chứa ma trận đề bài để kiểm tra xem lúc đột biến có bị trùng hay không
class originalComparison(Candidate):

    def __init__(self, values):
        self.values = values
        return

    # kiểm tra xem giá trị có bị trùng trong hàng không
    def isRowDuplicate(self, row, value):
        for column in range(0, NumDigits):
            if(self.values[row][column] == value):
                return True
        return False

    # kiểm tra xem giá trị có bị trùng trong cột không
    def isColumnDuplicate(self, column, value):
        for row in range(0, NumDigits):
            if(self.values[row][column] == value):
                return True
        return False

    # kiểm tra xem giá trị có bị trùng trong 3x3 không
    def isBlockDuplicate(self, row, column, value):
        i = 3*(int(row/3))
        j = 3*(int(column/3))

        if((self.values[i][j] == value)
           or (self.values[i][j+1] == value)
           or (self.values[i][j+2] == value)
           or (self.values[i+1][j] == value)
           or (self.values[i+1][j+1] == value)
           or (self.values[i+1][j+2] == value)
           or (self.values[i+2][j] == value)
           or (self.values[i+2][j+1] == value)
           or (self.values[i+2][j+2] == value)):
            return True
        else:
            return False


# class choiceParents, cho ra 2 cá thể để lai ghép với nhau
class choiceParents(object):
    """ The crossover function requires two parents to be selected from the population pool. The choiceParents class is used to do this.

    Two individuals are selected from the population pool and a random number in [0, 1] is chosen. If this number is less than the 'selection rate' (e.g. 0.85), then the fitter individual is selected; otherwise, the weaker one is selected.
    """

    def __init__(self):
        return

    def compete(self, candidates):
        """ Pick 2 random candidates from the population and get them to compete against each other. """
        c1 = candidates[random.randint(0, len(candidates)-1)]
        c2 = candidates[random.randint(0, len(candidates)-1)]
        f1 = c1.fitness
        f2 = c2.fitness

        # Find the fittest and the weakest.
        if(f1 > f2):
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        selection_rate = 0.85
        r = random.uniform(0, 1.1)
        while(r > 1):
            r = random.uniform(0, 1.1)
        if(r < selection_rate):
            return fittest
        else:
            return weakest


# tạo class lai ghép
class CycleCrossover(object):
    """ Crossover relates to the analogy of genes within each parent candidate mixing together in the hopes of creating a fitter child candidate. Cycle crossover is used here (see e.g. A. E. Eiben, J. E. Smith. Introduction to Evolutionary Computing. Springer, 2007). """

    def __init__(self):
        return

    def crossover(self, parent1, parent2, crossoverRate):
        # khởi tạo 2 thế hệ con
        child1 = Candidate()
        child2 = Candidate()

        # tạo bản sao từ thế hệ bố mẹ
        child1.values = numpy.copy(parent1.values)
        child1.fitness = parent1.fitness
        child2.values = numpy.copy(parent2.values)
        child2.fitness = parent2.fitness

        r = random.uniform(0, 1.1)
        while(r > 1):  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)

        # bắt đầu lai chéo
        if (r < crossoverRate):
            # chọn điểm giao nhau, ít nhất 1 hàng và nhiều nhất là NumDigits - 1 hàng
            crossoverPoint1 = random.randint(0, 8)
            crossoverPoint2 = random.randint(1, 9)
            while(crossoverPoint1 == crossoverPoint2):
                crossoverPoint1 = random.randint(0, 8)
                crossoverPoint2 = random.randint(1, 9)

            # crossoverPoint1 luôn luôn < crossoverPoint2
            if(crossoverPoint1 > crossoverPoint2):
                temp = crossoverPoint1
                crossoverPoint1 = crossoverPoint2
                crossoverPoint2 = temp

            # mỗi lần lặp là sẽ đổi 1 hàng bên child1 với 1 hàng bên child2
            for i in range(crossoverPoint1, crossoverPoint2):
                child1.values[i], child2.values[i] = self.crossoverRows(
                    child1.values[i], child2.values[i])

        return child1, child2

    # tiến hành hoán đổi giá trị 2 bên dựa vào vị trí đã chọn trong hàm crossover

    # hoán đổi dựa trên vị trí của hàng
    def crossoverRows(self, row, row2):
        childRow1 = numpy.zeros(NumDigits)
        childRow2 = numpy.zeros(NumDigits)

        # tạo mảng remaining, có giá trị từ 1 -> NumDigits
        remaining = [i for i in range(1, NumDigits+1)]

        cycle = 0

        # While child rows not complete...
        while((0 in childRow1) and (0 in childRow2)):
            if(cycle % 2 == 0):  # cycles chẵn
                # Assign next unused value.
                index = self.findUnused(row, remaining)
                start = row[index]
                remaining.remove(row[index])
                childRow1[index] = row[index]
                childRow2[index] = row2[index]
                next = row2[index]

                while(next != start):  # While cycle not done...
                    index = self.findValue(row, next)
                    childRow1[index] = row[index]
                    remaining.remove(row[index])
                    childRow2[index] = row2[index]
                    next = row2[index]

                cycle += 1

            else:  # cycle lẻ - flip values.
                index = self.findUnused(row, remaining)
                start = row[index]
                remaining.remove(row[index])
                childRow1[index] = row2[index]
                childRow2[index] = row[index]
                next = row2[index]

                while(next != start):  # While cycle not done...
                    index = self.findValue(row, next)
                    childRow1[index] = row2[index]
                    remaining.remove(row[index])
                    childRow2[index] = row[index]
                    next = row2[index]

                cycle += 1
        # print("\ncon 1: ", childRow1)
        # print("\ncon 2: ", childRow2)

        return childRow1, childRow2

    def findUnused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if(parent_row[i] in remaining):
                return i

    def findValue(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if(parent_row[i] == value):
                return i

    def swap2row(self, row, row2):
        temp = row
        row = row2
        row2 = temp
        return row, row2


class Sudoku(object):
    def __init__(self):
        self.original_Comparison = None
        return

    def load(self, path):
        # Load a file containing SUDOKU to solve.
        with open(path, "r") as f:
            values = numpy.loadtxt(f).astype(int)
            self.original_Comparison = originalComparison(values)
        print("INPUT\n", values)
        return

    def save(self, path, solution):
        # Save a configuration to a file.
        with open(path, "w") as f:
            numpy.savetxt(f, solution.values.reshape(
                NumDigits*NumDigits), fmt='%d')
        return

    # hàm xử lý
    def solve(self):
        size = 200  # kích thước quần thể
        sizeElite = int(0.6*size)  # số lượng cá thể ưu tú được giữ lại = 120
        NumGeneration = 999  # số thế hệ của quần thể
        NumMutate = 0  # đểm số lượng đột biến
        staleCount = 0  # đếm số thế hệ trong 1 vùng đã chọn
        prevFitness = 0

        # Xác định các biến được sử dụng để cập nhật đột biếnRate
        phi = 0  # để đếm số lần con giỏi hơn cha mẹ
        sigma = 1  # được sử dụng để cập nhật tỷ lệ đột biến
        mutationRate = 0.5  # tỉ lệ đột biến

        # quần thể ban đầu or tạo ra giống mới
        self.population = Population()
        self.population.seed(size, self.original_Comparison)

        # cho phép lặp 1000 thế hệ
        for generation in range(0, NumGeneration):
            print("Generation %d" % generation)

            # kiểm tra giải pháp
            bestFitness = 0.0
            bestSolution = self.original_Comparison
            # cho mỗi thế hệ, duyệt qua tất cả các ứng cử viên hoặc nhiễm sắc thể để kiểm tra giải pháp
            for c in range(0, size):
                fitness = self.population.candidates[c].fitness
                if(int(fitness) == 1):
                    print("Solution found at generation %d!" % generation)
                    print(self.population.candidates[c].values)
                    return self.population.candidates[c]

                # Find the best fitness.
                if(fitness > bestFitness):
                    bestFitness = fitness
                    bestSolution = self.population.candidates[c].values

            print("Best fitness: %f" % bestFitness)

            # Tạo quần thể tiếp theo.
            nextPopulation = []

            # Lựa chọn những người ưu tú (những ứng viên phù hợp nhất) và bảo tồn họ cho thế hệ sau.
            # 0.6*200=120 elites in new generation
            self.population.sort()
            elites = []
            for e in range(0, sizeElite):
                elite = Candidate()
                elite.values = numpy.copy(self.population.candidates[e].values)
                elites.append(elite)

            # Tạo phần còn lại của các ứng cử viên. 80 trẻ em, vì vậy chạy vòng lặp 40 lần
            for count in range(sizeElite, size, 2):
                # Chọn cha mẹ từ dân số thông qua a tournament.
                t = choiceParents()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)

                # lai chéo
                cc = CycleCrossover()
                child1, child2 = cc.crossover(
                    parent1, parent2, crossoverRate=1.0)

                # đột biến child1.
                child1.updateFitness()
                oldFitness = child1.fitness
                success = child1.mutate(mutationRate, self.original_Comparison)
                child1.updateFitness()
                if(success):
                    NumMutate += 1
                    # Được sử dụng để tính toán tỷ lệ thành công tương đối của các đột biến.
                    if(child1.fitness > oldFitness):
                        phi = phi + 1

                # đột biến child2.
                child2.updateFitness()
                oldFitness = child2.fitness
                success = child2.mutate(mutationRate, self.original_Comparison)
                child2.updateFitness()
                if(success):
                    NumMutate += 1
                    # Được sử dụng để tính toán tỷ lệ thành công tương đối của các đột biến.
                    if(child2.fitness > oldFitness):
                        phi = phi + 1

                # Thêm trẻ em vào dân số mới.
                nextPopulation.append(child1)
                nextPopulation.append(child2)

            # Thêm giới tinh hoa vào phần cuối của dân số. Chúng sẽ không bị ảnh hưởng bởi sự trao đổi chéo hoặc đột biến.
            for e in range(0, sizeElite):
                nextPopulation.append(elites[e])

            # Chọn thế hệ tiếp theo.
            self.population.candidates = nextPopulation
            self.population.updateFitness()

            # Tính tỷ lệ đột biến thích nghi mới (dựa trên quy tắc thành công 1/5 của Rechenberg). Điều này nhằm ngăn chặn quá nhiều đột biến khi thể lực tiến dần đến sự thống nhất.
            if(NumMutate == 0):
                phi = 0  # Avoid divide by zero.
            else:
                phi = phi / NumMutate

            if(phi > 0.2):
                sigma = sigma*0.998  # sigma giảm, ít đột biến hơn
            if(phi < 0.2):
                sigma = sigma/0.998  # sigma tăng, đột biến nhiều hơn
            mutationRate = abs(numpy.random.normal(
                loc=0.0, scale=sigma, size=None))
            while mutationRate > 1:
                mutationRate = abs(numpy.random.normal(
                    loc=0.0, scale=sigma, size=None))

            # Kiểm tra dân số cũ.
            self.population.sort()

            if generation == 0:
                prevFitness = bestFitness
                staleCount = 1

            elif prevFitness == bestFitness:
                staleCount += 1

            elif prevFitness != bestFitness:
                staleCount = 0
                prevFitness = bestFitness

            # Gieo hạt lại quần thể nếu 100 thế hệ đã qua mà hai ứng viên khỏe nhất luôn có cùng thể trạng.(không thể tạo ra cá thể khác biệt trong quần thể)
            if(staleCount >= 100):
                print("Dân số đã trở nên cũ kỹ. Đang làm mới dân số lại ...")
                self.population.seed(size, self.original_Comparison)
                staleCount = 0
                sigma = 1
                phi = 0
                mutations = 0
                mutationRate = 0.5

        print("Không tìm thấy giải pháp.", bestSolution)
        return None


s = Sudoku()
s.load("easy.txt")
solution = s.solve()
if(solution):
    s.save("solution.txt", solution)
