#!/usr/bin/env python

"""
This module is responsible for matrix manipulation and data processing
concerning key rate computation (BPSC and BPS).

This module is a part of the
"Examining noise models for high-dimensional Quantum
Key Distribution with subspace encoding" thesis.

@author Oskar Adam Valent
"""
from math import log2
from typing import Optional, Union, List

# Data aliases for prettier representation of matrices
DATA_MATRIX = List[Optional[int]]
PROB_MATRIX = List[Optional[float]]
GENERAL_MATRIX = List[Optional[Union[int, float]]]
NOISE_MATRIX = DATA_MATRIX


def print_matrix(d: int, matrix: GENERAL_MATRIX) -> None:
    """Prints matrix in human-readable format.

    @param d: The global dimension of matrix.
    @param matrix: The actual matrix.
    """
    for row in range(d):
        for col in range(d):
            print("{:^8}".format(str(matrix[row * d + col])), end="  ")
        print()


def print_2matrices(d: int, A: GENERAL_MATRIX, B: GENERAL_MATRIX) -> None:
    """Prints 2 matrices next ot each other in human-readable format.

    @param d: The global dimension of both matrices A and B.
    @param A: First matrix.
    @param B: Second matrix.
    """
    for i in range(d):
        for j in range(2 * d):
            if j == d:
                print("\t|\t", end="")
            if j >= d:
                print("{:^8}".format(str(B[i * d + (j - d)])), end="  ")
            else:
                print("{:^8}".format(str(A[i * d + j])), end="  ")
        print()
    print()


def subspace_sum(i: int, d: int, k: int, matrix: DATA_MATRIX) -> int:
    """Computes the sum of the subspace of a matrix.

    @param i: Index of cell in matrix where subspace begins.
    @param d: The global dimension of matrix.
    @param k: Dimension of subspace within matrix.
    @param matrix: Actual matrix being processed.
    @returns: The sum of clicks within specific subspace.
    """
    total = 0
    for _ in range(k):
        for j in range(k):
            total += matrix[i + j]
        # Shift to next row in the same subspace
        i += d

    return total


def all_subspaces_sum(d: int, k: int, matrix: DATA_MATRIX) -> int:
    """Computes the sum of all subspaces on diagonal in matrix.

    @param d: The global dimension of matrix.
    @param k: Dimension of subspaces on diagonal in matrix.
    @param matrix: The actual matrix being processed.
    @returns: Total sum of clicks in all subspaces.
    """
    total = 0
    for i in range(d // k):
        # Start index of each subspace
        index = i * (d * k + k)
        total += subspace_sum(index, d, k, matrix)
    return total


def probability_matrix(d: int, k: int, matrix: DATA_MATRIX) -> PROB_MATRIX:
    """Computes the matrix of probabilities that
       a cell detected click within its subspace.

    @param d: The global dimension of matrix.
    @param k: Dimension of subspaces on diagonal in matrix.
    @param matrix: Matrix being processed.
    @returns: The probability matrix
    """
    out_matrix = [Optional[int] for _ in range(d * d)]
    index = 0

    for _ in range(d // k):
        subspace_volume = subspace_sum(index, d, k, matrix)
        for _ in range(k):
            for i in range(k):
                cell = matrix[index + i]

                if subspace_volume == 0:
                    # avoid division by zero in case of empty subspace
                    out_matrix[index + i] = 0
                else:
                    # compute cell probability that it detected click
                    out_matrix[index + i] = round(cell / subspace_volume, 6)
            # Move to the next row of actual subspace
            index += d
        # Move to the next subspace
        index += k

    return out_matrix


def d8k224_probability_matrix(matrix: DATA_MATRIX) -> PROB_MATRIX:
    """Computes matrix of probabilities that cell detected click
       in the special case of \"Simple QKD network\" where matrix has
       subspaces with dimensions 2, 2  and 4.

    @param matrix: QKD network compound matrix as input data.
    @returns: Probability matrix for case of QKD network.
    """
    prob_matrix = [Optional[float] for _ in matrix]
    subspaces_dimension = [2, 2, 4]
    index = 0

    for k in subspaces_dimension:
        subspace_volume = subspace_sum(index, 8, k, matrix)
        for _ in range(k):
            for i in range(k):
                cell = matrix[index + i]

                if subspace_volume == 0:
                    # avoid division by zero in case of empty subspaces
                    prob_matrix[index + i] = 0
                else:
                    prob_matrix[index + i] = round(cell / subspace_volume, 6)
            # move to the next row of actual subspace
            index += 8
        # move to the next subspace
        index += k

    return prob_matrix


def d8k224_keyrate_per_subspace(comp_matrix: PROB_MATRIX, fourier_matrix: PROB_MATRIX) -> List[float]:
    """Computes key rate for each subspace in \"simple QKD network\"
       scenario.

    @param comp_matrix: Computational matrix.
    @param fourier_matrix: Fourier matrix.
    @returns: List of subspace key rates.
    """
    subspace_keyrates = []
    log_diagonal_c = 0
    log_diagonal_f = 0
    subspace_dimensions = [2, 2, 4]
    cell_index_per_subspace = -18

    for i in range(3):
        cell_index_per_subspace += 18
        k = subspace_dimensions[i]
        value = log2(k)

        # if subspace_sum(cell_index_per_subspace, 8, k, fourier_matrix) == 0 or \
        #    subspace_sum(cell_index_per_subspace, 8, k, comp_matrix) == 0:
        #     subspace_keyrates.append(0.0)
        #     continue

        for j in range(k):
            # Start index of each <k> possible diagonals within subspace
            index = j + i * (8 * 2 + 2)
            diagonal_c = diagonal_sum(index, 8, k, comp_matrix)
            diagonal_f = diagonal_sum(index, 8, k, fourier_matrix)

            if diagonal_c != 0:
                # avoid undefined log for empty diagonals
                log_diagonal_c = log2(diagonal_c)
            else:
                diagonal_c = -1
                log_diagonal_c = log2(k)/float(k)

            if diagonal_f != 0:
                # avoid undefined log for empty diagonal
                log_diagonal_f = log2(diagonal_f)
            else:
                diagonal_f = -1
                log_diagonal_f = log2(k)/float(k)

            # compute key rate for subspace according to formula
            value -= -(diagonal_c * log_diagonal_c + diagonal_f * log_diagonal_f)

        subspace_keyrates.append(round(value, 6))

    return subspace_keyrates


def diagonal_sum(index: int, d: int, k: int, matrix: PROB_MATRIX) -> float:
    """Computes sum of cells on specified (non-classical) diagonal
       in matrix.

    @param index: Start index of cell in diagonal.
    @param d: The global dimension of probability matrix.
    @param k: Dimension of subspace within matrix.
    @param matrix: The actual matrix being processed.
    """
    total = 0
    for i in range(k):
        # if index is pointing out of subspace, move to the
        # next cell that would follow on non-classical diagonal
        if i != 0 and index % k == 0:
            index -= k

        total += matrix[index]
        # move to the next cell on diagonal
        index += (d + 1)

    return total


def key_rate_per_subspace(d: int, k: int, c_matrix: PROB_MATRIX, f_matrix: PROB_MATRIX) -> List[float]:
    """Computes key rate for each subspace in symmetric scenarios.

    @param d: The global dimension of input matrices.
    @param k: Dimension of subspaces within input matrices.
    @param c_matrix: Computational matrix as input data.
    @param f_matrix: Fourier matrix as input data.
    @returns: List of subspace key rates.
    """
    subspace_keyrates = []
    log_diagonal_c = 0
    log_diagonal_f = 0

    for i in range(d // k):
        value = log2(k)
        for j in range(k):
            # Start index of each <k> possible diagonals within subspace
            index = j + i * (d * k + k)
            diagonal_comp = diagonal_sum(index, d, k, c_matrix)
            diagonal_fourier = diagonal_sum(index, d, k, f_matrix)

            if diagonal_comp != 0:
                # avoid undefined log for empty diagonals
                log_diagonal_c = log2(diagonal_comp)
            else:
                diagonal_comp = -1
                log_diagonal_c = log2(k)/float(k)

            if diagonal_fourier != 0:
                # avoid undefined log for empty diagonals
                log_diagonal_f = log2(diagonal_fourier)
            else:
                diagonal_fourier = -1
                log_diagonal_f = log2(k)/float(k)

            # compute key rate for subspace according to formula
            value -= -(diagonal_comp * log_diagonal_c + diagonal_fourier * log_diagonal_f)
        subspace_keyrates.append(round(value, 6))

    return subspace_keyrates


def average_key_rate(d: int, k: int, matrix: DATA_MATRIX, subspace_keyrates: List[float]) -> float:
    """Computes average key rate from subspace key rates and
       computational matrix.

    @param d: The global dimension of input computational matrix
    @param k: Dimension of subspaces within matrix.
    @param matrix: The computational matrix as input data.
    @param subspace_keyrates: Subspace key rates associated with matrix.
    @returns: Average key rate for specific QKD instance.
    """
    subspace_volumes = []
    avg_key_rate = 0
    subspace_index = 0

    # compute sum of clicks throughout all subspaces
    for i in range(d // k):
        # start cell index of i-th subspace
        index = i * (d * k + k)
        subspace_volumes.append(subspace_sum(index, d, k, matrix))

    subspace_volumes_total = sum(subspace_volumes)

    # compute average keyrate
    for volume in subspace_volumes:
        if volume == 0:
            avg_key_rate += 0
        else:
            avg_key_rate += volume / subspace_volumes_total * subspace_keyrates[subspace_index]
        subspace_index += 1

    return round(avg_key_rate, 6)


def d8k224_average_keyrate(matrix: DATA_MATRIX, subspace_keyrates: List[float]) -> float:
    """Computes average key rate for case \"simple QKD network\".

    @param matrix: Computational matrix as input data.
    @param subspace_keyrates: Subspace key rates associated with matrix.
    @returns: Average key rate for \"simple QKD network\" instance.
    """
    subspace_volumes = []
    avg_keyrate = 0
    subspace_index = 0

    # calculate sum of clicks of all subspaces within matrix
    # and make list of total clicks throughout subspaces
    for k in [2, 2, 4]:
        subspace_volumes.append(subspace_sum(subspace_index, 8, k, matrix))
        subspace_index += 18

    subspace_volumes_total = sum(subspace_volumes)

    # calculate average key rate according to formula
    for subspace_index, volume in enumerate(subspace_volumes):
        if subspace_volumes_total <= 100:
            avg_keyrate += 0
        else:
            avg_keyrate += volume / subspace_volumes_total * subspace_keyrates[subspace_index]

    return round(avg_keyrate, 6)


def d8k224_subspaces_sum(matrix: DATA_MATRIX):
    """Computes sum of clicks in all subspaces within matrix
       in case of \"Simple QKD network\" scenario.

    @param matrix: The matrix being processed.
    """
    total = 0
    subspace_start_index = 0

    for k in [2, 2, 4]:
        total += subspace_sum(subspace_start_index, 8, k, matrix)
        subspace_start_index += 18

    return total


def subspace_probabilities(d: int, k: int, matrix: DATA_MATRIX) -> List[float]:
    """Computes probabilities of subspaces that click
       would be detected in subspace.

    @param d: The global dimension of input matrix.
    @param k: Dimension of subspaces within matrix.
    @param matrix: The matrix being processed.
    @returns: List of probabilities for each subspace that click
              would hit that particular subspace.
    """
    subspace_volumes = []
    probabilities = []

    # calculate sum of clicks of all subspaces within matrix
    # and make list of total clicks throughout subspaces
    for i in range(d // k):
        subspace_start_index = i * (d * k + k)
        subspace_volumes.append(subspace_sum(subspace_start_index, d, k, matrix))

    volumes_sum = sum(subspace_volumes)

    # compute probabilities for each subspace
    for volume in subspace_volumes:
        probabilities.append(round(volume / volumes_sum, 6))

    return probabilities


def kbit_per_second(d: int, k: int, matrix: DATA_MATRIX, avg_key_rate: float, duration: float = 25.0) -> float:
    """Computes key rate in kbps.

    @param d: The global dimension of input computational matrix.
    @param k: Dimension of subspaces within matrix.
    @param matrix: Computational matrix being processed.
    @param avg_key_rate: Average key rate of actual QKD instance.
    @param duration: Duration of the experiment.
    @returns: Key rate in kbps.
    """
    avg_photon_injected = all_subspaces_sum(d, k, matrix) / duration

    return round(avg_key_rate * avg_photon_injected, 6)


def show_keyrate_computations(d: int, k: int, f_matrix: DATA_MATRIX, c_matrix: DATA_MATRIX) -> None:
    """Shows process of computation of key rate in terminal.

    @param d: The global dimension of input matrices.
    @param k: Dimension of subspaces within matrix.
    @param f_matrix: Fourier matrix as input data.
    @param c_matrix: Computational matrix as input data.
    """
    print("Fourier probability matrix: ")
    f_prob_matrix = probability_matrix(d, k, f_matrix)
    print_matrix(d, f_prob_matrix)

    print("Computational probability matrix: ")
    c_prob_matrix = probability_matrix(d, k, c_matrix)
    print_matrix(d, c_prob_matrix)

    print("Subspace probability: ")
    probs = subspace_probabilities(d, k, f_matrix)
    for i in range(d // k):
        print("\tS", i + 1, ":", probs[i])

    print("Key rate(s) per coincidence: ")
    krpc = key_rate_per_subspace(d, k, c_prob_matrix, f_prob_matrix)
    for i in range(d // k):
        print("\tS", i + 1, ":", krpc[i])

    kavg = average_key_rate(d, k, f_matrix, krpc)
    print("Average key rate: [", kavg, ']')

    print("KBPS: [", kbit_per_second(d, k, f_matrix, kavg), ']')
    print('*' * (d * 8 + (d - 1) * 2))
    print()


def show_computation(d: int, k: int, p: float, f_matrix: DATA_MATRIX, c_matrix: DATA_MATRIX) -> None:
    """Shows input data, setting and process of computation
    of key rate in terminal.

    @param d: The global dimension of input matrices.
    @param k: Dimension of subspaces within matrix.
    @param p: Noise parameter.
    @param f_matrix: Fourier matrix as input data.
    @param c_matrix: Computational matrix as input data.
    """
    print("Noise parameter p: [", p, ']')
    print("Computational basis dimension d: [", d, ']')
    print("Subspace dimension k: [", k, "]\n")

    print("Fourier matrix: ")
    print_matrix(d, f_matrix)

    print("Computational matrix: ")
    print_matrix(d, c_matrix)

    show_keyrate_computations(d, k, f_matrix, c_matrix)

