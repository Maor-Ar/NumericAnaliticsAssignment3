"""
 * Authors: Maor Arnon (ID: 205974553) and Neriya Zudi (ID:207073545)
 * Emails: maorar1@ac.sce.ac.il    neriyazudi@Gmail.com
 * Department of Computer Engineering - Assignment 3 - Numeric Analytics
"""


def PrintMatrix(matrix):
    """
    Matrix Printing Function
    :param matrix: Matrix nxn
    """
    for line in matrix:
        print('  '.join(map(str, line)))


def MaxNorm(matrix):
    """
    Function for calculating the max-norm of a matrix
    :param matrix: Matrix nxn
    :return:max-norm of a matrix
    """
    max_norm = 0
    for i in range(len(matrix)):
        norm = 0
        for j in range(len(matrix)):
            # Sum of organs per line with absolute value
            norm += abs(matrix[i][j])
        # Maximum row amount
        if norm > max_norm:
            max_norm = norm

    return max_norm


def Determinant(matrix, mul):
    """
    Recursive function for determinant calculation
    :param matrix: Matrix nxn
    :param mul: The double number
    :return: determinant of matrix
    """
    width = len(matrix)
    # Stop Conditions
    if width == 1:
        return mul * matrix[0][0]
    else:
        sign = -1
        det = 0
        for i in range(width):
            m = []
            for j in range(1, width):
                buff = []
                for k in range(width):
                    if k != i:
                        buff.append(matrix[j][k])
                m.append(buff)
            # Change the sign of the multiply number
            sign *= -1
            #  Recursive call for determinant calculation
            det = det + mul * Determinant(m, sign * matrix[0][i])
    return det


def MakeIMatrix(cols, rows):
    # Initialize a identity matrix
    return [[1 if x == y else 0 for y in range(cols)] for x in range(rows)]


def MultiplyMatrix(matrixA, matrixB):
    """
    Function for multiplying 2 matrices
    :param matrixA: Matrix nxn
    :param matrixB: Matrix nxn
    :return: Multiplication between 2 matrices
    """
    # result matrix initialized as singularity matrix
    result = [[0 for y in range(len(matrixB[0]))] for x in range(len(matrixA))]
    for i in range(len(matrixA)):
        # iterate through columns of Y
        for j in range(len(matrixB[0])):
            # iterate through rows of Y
            for k in range(len(matrixB)):
                result[i][j] += matrixA[i][k] * matrixB[k][j]
    return result


def InverseMatrix(matrix, vector):
    """
    Function for calculating an inverse matrix
    :param matrix:  Matrix nxn
    :return: Inverse matrix
    """
    # Unveri reversible matrix
    if Determinant(matrix, 1) == 0:
        print("Error,Singular Matrix\n")
        return
    # result matrix initialized as singularity matrix
    result = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # turn the pivot into 1 (make elementary matrix and multiply with the result matrix )
        # pivoting process
        matrix, vector = RowXchange(matrix, vector)
        elementary = MakeIMatrix(len(matrix[0]), len(matrix))
        elementary[i][i] = 1 / matrix[i][i]
        result = MultiplyMatrix(elementary, result)
        matrix = MultiplyMatrix(elementary, matrix)
        # make elementary loop to iterate for each row and subtracrt the number below (specific) pivot to zero  (make
        # elementary matrix and multiply with the result matrix )
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i])
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)

    # after finishing with the lower part of the matrix subtract the numbers above the pivot with elementary for loop
    # (make elementary matrix and multiply with the result matrix )
    for i in range(len(matrix[0]) - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i])
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)

    return result

def InverseMatrix(matrix):
    """
    Function for calculating an inverse matrix
    :param matrix:  Matrix nxn
    :return: Inverse matrix
    """
    vector=[0]*len(matrix)
    # Unveri reversible matrix
    if Determinant(matrix, 1) == 0:
        print("Error,Singular Matrix\n")
        return
    # result matrix initialized as singularity matrix
    result = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # turn the pivot into 1 (make elementary matrix and multiply with the result matrix )
        # pivoting process
        matrix, vector = RowXchange(matrix, vector)
        elementary = MakeIMatrix(len(matrix[0]), len(matrix))
        elementary[i][i] = 1 / matrix[i][i]
        result = MultiplyMatrix(elementary, result)
        matrix = MultiplyMatrix(elementary, matrix)
        # make elementary loop to iterate for each row and subtracrt the number below (specific) pivot to zero  (make
        # elementary matrix and multiply with the result matrix )
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i])
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)

    # after finishing with the lower part of the matrix subtract the numbers above the pivot with elementary for loop
    # (make elementary matrix and multiply with the result matrix )
    for i in range(len(matrix[0]) - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i])
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)

    return matrix

def RowXchange(matrix, vector):
    """
    Function for replacing rows with both a matrix and a vector
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Replace rows after a pivoting process
    """

    for i in range(len(matrix)):
        max = abs(matrix[i][i])
        for j in range(i, len(matrix)):
            # The pivot member is the maximum in each column
            if abs(matrix[j][i]) > max:
                temp = matrix[j]
                temp_b = vector[j]
                matrix[j] = matrix[i]
                vector[j] = vector[i]
                matrix[i] = temp
                vector[i] = temp_b
                max = abs(matrix[i][i])

    return [matrix, vector]


def CheckDominantDiagonal(matrix):
    for i in range(len(matrix)):
        sum = 0
        for j in range(len(matrix)):
            if i != j:
                sum += abs(matrix[i][j])
        if abs(matrix[i][i]) < sum:
            return False
    return True


# def DominantDiagonalFix(matrix):
#     #Check if we have a dominant for each column
#     dom = []
#     result = list()
#     for i in range(len(matrix)):
#         for j in range(len(matrix[0])):
#             if matrix[i][j] > sum(map(abs,matrix[i])):
#                 dom[i]=j
#     for i in range(len(matrix)):
#         if i not in dom:
#             print("Couldn't find dominant diagonal.")
#             return matrix
#     for i,j in enumerate(dom):
#         if i not in dom:
#             return False
#     return True

def minusMatrix(matrix):
    return [[-i for i in j] for j in matrix]


def matrixAddition(matrixA, matrixB):
    return [[a + b for (a, b) in zip(i, j)] for (i, j) in zip(matrixA, matrixB)]


def matrixDLUdissasembly(matrix):
    D, L, U = list(), list(), list()
    for x, row in enumerate(matrix):
        D.append(list()), L.append(list()), U.append(list())
        for y, value in enumerate(row):
            if x == y:
                D[x].append(value), L[x].append(0), U[x].append(0)
            elif x < y:
                D[x].append(0), L[x].append(0), U[x].append(value)
            elif x > y:
                D[x].append(0), L[x].append(value), U[x].append(0)
    return D, L, U

def YakobiG(matrix):
    D, L, U = matrixDLUdissasembly(matrix)
    return MultiplyMatrix(minusMatrix(InverseMatrix(D)), InverseMatrix(matrixAddition(L, U)))

def YakobiH(matrix):
    D, L, U = matrixDLUdissasembly(matrix)
    return InverseMatrix(D)

def GeusZaidelG(matrix):
    D, L, U = matrixDLUdissasembly(matrix)
    return MultiplyMatrix(minusMatrix(InverseMatrix(matrixAddition(L, D))), U)

def GeusZaidelH(matrix):
    D, L, U = matrixDLUdissasembly(matrix)
    return InverseMatrix(matrixAddition(L,D))

def CheckYakobiGnorm(matrix):
    return 1 > MaxNorm(YakobiG(matrix))

def CheckGeusZaidelGnorm(matrix):
    return 1 > MaxNorm(GeusZaidelG(matrix))

def solveByYakobi(matrix,b,epsilon):
    itteration=0
    G = YakobiG(matrix)
    H = YakobiH(matrix)
    X = [0] * len(matrix)
    newX = [epsilon] * len(matrix)
    first_iteration = False
    while abs(sum(X)-sum(newX))>3*epsilon or itteration > 100:
        if not first_iteration:
            X = newX
            first_iteration = False
        newX = matrixAddition(MultiplyMatrix(G,X),MultiplyMatrix(H,b))
        itteration+=1
        print(itteration+") "+X)
    return newX

matrixA = [[1, 0, -1], [2, 5, 1], [3, 4, 9]]
matrixB = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
b = [10,20,30]
# for i in matrixDLUdissasembly(matrixA):
#     print()
#     PrintMatrix(i)
if CheckYakobiGnorm(matrixB):
    print(solveByYakobi(matrixB,b,0.001))
else:
    print("Fkkk")
a=[[0,0,0],[2,0,0],[0,4,0]]
b=[[0,2,0],[0,0,4],[0,0,0]]

# PrintMatrix(matrixA)
# PrintMatrix(matrixAddition(matrixA,matrixB))
# PrintMatrix(minusMatrix(matrixA))
# vectorb = [[10], [20], [30]]
# PrintMatrix(matrixA)
# if CheckDominantDiagonal(matrixA):
#     print("Diagonal  dominant")
# else:
#     print("Diagonal not dominant")
