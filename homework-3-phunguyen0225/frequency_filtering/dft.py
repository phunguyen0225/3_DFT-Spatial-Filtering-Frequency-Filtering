# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import numpy as np
import itertools
import math


class Dft:
    def __init__(self):
        pass

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        """
        for row, u in itertools.product(range(len(matrix[0])-1), range(len(matrix[0])-1)):
            for col, v in itertools.product(range(len(matrix[1])-1), range(len(matrix[1])-1)):
                forward[u, v] = matrix[row, col] * (math.cos(((2*math.pi)/N) * (u*row + v*col)) -
                                                    1j * math.sin((2*math.pi)/N) * (u*row + v*col))
        """
        nRow = matrix.shape[0]
        nCol = matrix.shape[1]

        M = nRow
        N = nCol

        forward = np.zeros((nRow, nCol), dtype=complex)
        for u in range(nRow):
            for v in range(nCol):
                total = 0
                for i in range(nRow):
                    for j in range(nCol):
                        perPeriod = (u * i / M) + (v * j / N)
                        e = -2j * np.pi * perPeriod
                        value = matrix[i, j] * np.exp(e)
                        total += value
                forward[u, v] = total

        return forward

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        You can implement the inverse transform formula with or without the normalizing factor.
        Both formulas are accepted.
        takes as input:
        matrix: a 2d matrix (DFT) usually complex
        returns a complex matrix representing the inverse fourier transform"""
        """
        for row, u in itertools.product(range(len(matrix[0])-1), range(0, len(matrix[0])-1)):
            for col, v in itertools.product(range(len(matrix[1])-1), range(0, len(matrix[1])-1)):
                inverse[row, col] = matrix[u, v] * (math.cos(((2*math.pi)/N) * (u*row + v*col)) +
                                                    1j * math.sin((2*math.pi)/N) * (u*row + v*col))
        """
        nRow = matrix.shape[0]
        nCol = matrix.shape[1]

        M = nRow
        N = nCol

        inverse = np.zeros((nRow, nCol), dtype=complex)

        for i in range(nRow):
            for j in range(nCol):
                total = 0
                for u in range(nRow):
                    for v in range(nCol):
                        perPeriod = (i * u / M) + (j * v / N)
                        e = 2j * np.pi * perPeriod
                        value = matrix[u, v] * np.exp(e)
                        total += value
                total = total / (M * N)
                inverse[i, j] = total

        return inverse

    def magnitude(self, matrix):
        """Computes the magnitude of the input matrix (iDFT)
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the complex matrix"""
        nRow = matrix.shape[0]
        nCol = matrix.shape[1]

        mag_matrix = np.zeros((nRow, nCol), dtype=int)
        magnitude_ = 0
        for u in range(nRow):
            for v in range(nCol):
                mag_matrix[u, v] = math.sqrt(
                    np.real(matrix[u, v])**2 + np.imag(matrix[u, v] ** 2))
        return mag_matrix
