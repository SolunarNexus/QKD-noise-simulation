#!/usr/bin/env python

"""
This module contains functions for performing noise models simulations
and generating result graphs for the purpose of thesis.

This module is a part of the
"Examining noise models for high-dimensional Quantum
Key Distribution with subspace encoding" thesis.

@author Oskar Adam Valent
"""
import bisect
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keyrate_computation import *
from math import floor, sqrt, pow
from adjustText import adjust_text


def compute_out(dB: float, IN: int) -> int:
    """Computes the number of photons that were sent to Bob
    considering the channel loss.

    @param dB: Decibels as the measure of channel loss.
    @param IN: Number of photons detected in Alice's lab.
    @returns: Number of photons sent to Bob minus channel loss.
    """
    return floor(IN / pow(10, dB / 10))


def compute_loss(dB: float, IN: int) -> int:
    """Computes the number of lost photons during transmission.

    @param dB: Decibels as the measure of channel loss.
    @param IN: Number of photons detected in Alice's lab.
    @returns: Number of lost photons during theoretical transmission.
    """
    return IN - compute_out(dB, IN)


def loss_matrix(matrix: DATA_MATRIX, dB: float, loss: int = None) -> DATA_MATRIX:
    """Stochastically erases clicks from input matrix and thus
    simulates channel loss.

    @param matrix: Input matrix that will experience channel loss.
    @param dB: Decibels as the measure of channel loss.
    @param loss: Optional pre-determined amount of lost photons.
    @returns: Input matrix with erased lost photons.
    """
    if loss is None:
        loss = compute_loss(dB, sum(matrix))

    loss_mtrx = list(matrix)
    accumulated_sums = []
    total = 0

    # Get total sum of all cells and create a list of partial sums
    # of first 1...n cells, where n = {1, 2, ... , count of all cells}.
    for x in matrix:
        total += x
        accumulated_sums.append(total)

    # Stochastically erase clicks from the input matrix
    while loss > 0:
        choice = random.randint(0, total)
        index = bisect.bisect_left(accumulated_sums, choice)

        # erase only if the cell contains non-zero amount of clicks
        if loss_mtrx[index] > 0:
            loss_mtrx[index] -= 1
            loss -= 1

    return loss_mtrx


def coincidence_count(dS: int, tau: float = 5e-9, duration: float = 25.0) -> int:
    """Computes the number of extra coincidences (C) during
    the whole experiment according to formula C = 2 * d² * S² * τ * 25.

    @param dS: Single count rate * global dimension (d).
    @param tau: Coincidence window.
    @param duration: Duration of the experiment.
    @returns: Number of extra coincidences during the
    theoretical experiment.
    """
    return floor(2 * dS * dS * tau * duration)


def single_channel_count(C: int, d: int, tau: float = 5e-9, duration: float = 25.0) -> int:
    """Computes the single count rate (S) out of given
    extra coincidence rate (C).

    @param C: Extra coincidence rate.
    @param d: Global dimension.
    @param tau: Coincidence window.
    @param duration: Duration of the experiment.
    @returns: Single count rate (S).
    """
    return floor(sqrt(C / (2 * d ** 2 * tau * duration)))


def matrix_add(k: int, data: DATA_MATRIX, noise: NOISE_MATRIX, index: int = None, d: int = 0) -> DATA_MATRIX:
    """Classical form of matrix addition with a minor constraint - only
    noise matrices are added to data matrices.

    @param k: Dimension of subspaces within input matrix.
    @param data: Input data matrix.
    @param noise: Input noise matrix.
    @param index: Index of cell at which addition begins.
    @param d: Optional global dimension of matrix (qkd network).
    @returns: New matrix with added noise clicks.
    """
    result_matrix = [x for x in data]

    for i in range(k):
        # index into matrices
        idx = i * k
        for j in range(k):
            if index is None:
                result_matrix[idx + j] = data[idx + j] + noise[idx + j]
            else:
                result_matrix[index + j] = data[index + j] + noise[idx + j]

        if index is not None:
            index += d

    return result_matrix


def noise_matrix(d: int, C: int) -> NOISE_MATRIX:
    """Creates a noise matrix containing extra coincidences randomly
    spread across the whole matrix.

    @param d: The global dimension of noise matrix.
    @param C: Extra coincidences.
    @returns: Noise matrix.
    """
    noise = [0 for _ in range(d * d)]
    # create list of random indices with length of C in range <0, d*d)
    random_indices = random.choices(range(d * d), k=C)

    for index in random_indices:
        noise[index] += 1

    return noise


def marker_xcoord_detector_noise(d: int, total_coincidences: int) -> float:
    """Computes the x-coordinate for marker denoting result from the
    actual experiment in case of detector noise.

    @param d: The global dimension used in the setting.
    @param total_coincidences: Extra coincidence count rate (C).
    @returns: x-coordinate for experimental marker (S).
    """
    return single_channel_count(total_coincidences, d)


# def marker_xcoord_detector_noise_bps(d: int, total_coincidence: int) -> float:
#     return single_channel_count(total_coincidence, d)


def marker_xcoord_channel_noise(d: int, total_coincidences: int) -> float:
    """Computes the x-coordinate for marker denoting results from the
    actual experiment in case of channel noise.

    @param d: The global dimension used in the setting.
    @param total_coincidences: Extra coincidence count rate (C).
    @returns: x-coordinate for experimental marker (d*S)"""
    return d * single_channel_count(total_coincidences, d)


#
# def marker_xcoord_channel_noise_bps(d: int, total_coincidence: int) -> float:
#     return d * single_channel_count(total_coincidence, d)


def marker_xcoord_isotropic_noise(d: int, total_coincidences: int) -> float:
    """Computes the x-coordinate for marker denoting results from the
    actual experiment in case of isotropic noise.

    @param d: The global dimension used in the setting.
    @param total_coincidences: Extra coincidence count rate (C).
    @returns: x-coordinate for experimental marker (C/d/sec)"""
    return total_coincidences / (d * 25.0)


# def marker_xcoord_isotropic_noise_bps(d: int, total_coincidence: int):
#     return total_coincidence / (d * 25.0)


def put_markers(figure, xcoord_function, global_dims: List[int], subspace_dims: List[int],
                extra_coincidences: List[List[int]], keyrates: List[List[float]],
                add_labels: bool = False):
    """Draws markers for each setting and each noise level (p)
    into figure. Markers denote results obtained in the actual
    experiment and helps to show accuracy of graphs.

    @param figure: Graph object.
    @param xcoord_function: Function to compute the x-coordinate
    of the marker
    @param global_dims: List of global dimensions of settings.
    @param subspace_dims: List of subspace dimensions of settings.
    @param extra_coincidences: Extra coincidences count rates
    for each setting and each noise level.
    @param keyrates: Key rates for each setting and each noise level.
    @param add_labels: Optionally draws text labels into graphs.
    @returns: Modified graph with drawn markers.
    """
    noise_colors = ['xkcd:green', 'xkcd:lime green', 'xkcd:yellow', 'xkcd:orange', 'xkcd:red']
    # markers shapes (square, triangle, circle, diamond, plus)
    markers = ['s', 'v', 'o', 'D', 'P']
    mark_labels = ["p=0", "p=0.025", "p=0.075", "p=0.15", "p=0.3"]
    instance_idx = 0
    texts = []

    for instance in extra_coincidences:
        marker = markers[instance_idx]

        for setting_idx in range(len(instance)):
            color = noise_colors[setting_idx]
            d = global_dims[instance_idx]
            k = subspace_dims[instance_idx]
            C = extra_coincidences[instance_idx][setting_idx]
            keyrate = keyrates[instance_idx][setting_idx]
            # y-coordinate is always keyrate value
            x, y = xcoord_function(d, C), keyrate

            figure.plot(x, y, color=color, marker=marker, markeredgecolor='k', markeredgewidth=0.5)

            if add_labels:
                annotation = 'd' + str(d) + 'k' + str(k)
                texts.append(plt.text(x, y, annotation, fontsize=12))

        instance_idx += 1

    if add_labels:
        adjust_text(texts, expand_text=(1.2, 1.2), expand_points=(1.2, 1.2),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", lw=1))

    handles, labels = figure.get_legend_handles_labels()

    for i in range(len(mark_labels)):
        handles.append(mpatches.Patch(color=noise_colors[i], label=mark_labels[i]))

    plt.legend(handles=handles)

    return plt.gca()


def perform_symmetric_simulation(C: int, d: int, k: int, f_matrix: DATA_MATRIX,
                                 c_matrix: DATA_MATRIX, is_bpsc: bool = True) -> float:
    """Performs simulation of symmetric noise models.

    @param C: Extra coincidence count rate.
    @param d: The global dimension of input matrices.
    @param k: Dimension of subspaces within input matrices.
    @param f_matrix: Input Fourier matrix.
    @param c_matrix: Input computational matrix.
    @param is_bpsc: Switch between computation of BPSC and BPS.
    @returns: Key rate (BPSC or BPS).
    """
    f_matrix_noisy = matrix_add(d, f_matrix, noise_matrix(d, C))
    c_matrix_noisy = matrix_add(d, c_matrix, noise_matrix(d, C))

    f_probability_matrix = probability_matrix(d, k, f_matrix_noisy)
    c_probability_matrix = probability_matrix(d, k, c_matrix_noisy)

    keyrate_values = key_rate_per_subspace(d, k, c_probability_matrix, f_probability_matrix)

    average_keyrate = average_key_rate(d, k, f_matrix_noisy, keyrate_values)

    if is_bpsc:
        return average_keyrate

    return average_keyrate * all_subspaces_sum(d, k, f_matrix_noisy) / 25.0


def perform_asymmetric_simulation(dB: float, d: int, k: int, f_matrix: DATA_MATRIX, c_matrix: DATA_MATRIX,
                                  is_bpsc: bool = True, isolated_source: bool = False,
                                  C_f: int = 0, C_c: int = 0) -> float:
    """Performs simulation of transmission loss or isolated
    entanglement scenarios.

    @param dB: Decibels as the measure of lost photons.
    @param d: The global dimension of input matrices.
    @param k: Dimension of subspaces within input matrices.
    @param f_matrix: Input Fourier matrix.
    @param c_matrix: Input computational matrix.
    @param is_bpsc: Switch between computation of BPSC and BPS.
    @param isolated_source: Compute isolated entanglement source
    scenario, transmission loss otherwise.
    @param C_f: Extra coincidence count rate introduced to f_matrix.
    @param C_c: Extra coincidence count rate introduced to c_matrix.
    @returns: Key rate (BPSC or BPS).
    """
    f_matrix_modified = loss_matrix(f_matrix, dB)
    c_matrix_modified = loss_matrix(c_matrix, dB)

    if isolated_source:
        f_matrix_modified = matrix_add(d, f_matrix_modified, noise_matrix(d, C_f))
        c_matrix_modified = matrix_add(d, c_matrix_modified, noise_matrix(d, C_c))

    f_probability_matrix = probability_matrix(d, k, f_matrix_modified)
    c_probability_matrix = probability_matrix(d, k, c_matrix_modified)

    keyrate_values = key_rate_per_subspace(d, k, c_probability_matrix, f_probability_matrix)

    average_keyrate = average_key_rate(d, k, f_matrix_modified, keyrate_values)

    if is_bpsc:
        return average_keyrate

    return average_keyrate * all_subspaces_sum(d, k, f_matrix_modified) / 25.0


def detector_noise_fun(x: int, d: int, k: int, f_matrix: DATA_MATRIX, c_matrix: DATA_MATRIX,
                       is_bpsc: bool) -> float:
    """Helper function to compute y-coordinate (key rate) in case of
    detector noise scenario.

    @param x: Noise parameter (S/sec).
    @param d: The global dimension of input matrices.
    @param k: Dimension of subspaces within input matrices.
    @param f_matrix: Input Fourier matrix.
    @param c_matrix: Input computational matrix.
    @param is_bpsc: Switch between computing BPSC and BPS keyrate.
    @returns: Key rate (BPSC or BPS).
    """
    C = coincidence_count(d * x)

    # average key rate per photon
    return perform_symmetric_simulation(C, d, k, f_matrix, c_matrix, is_bpsc=is_bpsc)


def channel_noise_fun(x: int, d: int, k: int, f_matrix: DATA_MATRIX, c_matrix: DATA_MATRIX,
                      is_bpsc: bool) -> float:
    """Helper function to compute y-coordinate (key rate) in case of
    channel noise scenario.

    @param x: Noise parameter (d*S/sec).
    @param d: The global dimension of input matrices.
    @param k: Dimension of subspaces within input matrices.
    @param f_matrix: Input Fourier matrix.
    @param c_matrix: Input computational matrix.
    @param is_bpsc: Switch between computing BPSC and BPS keyrate.
    @returns: Key rate (BPSC or BPS).
    """
    C = coincidence_count(x)

    # average key rate per photon
    return perform_symmetric_simulation(C, d, k, f_matrix, c_matrix, is_bpsc=is_bpsc)


def isotropic_noise_fun(x: int, d: int, k: int, f_matrix: DATA_MATRIX, c_matrix: DATA_MATRIX,
                        is_bpsc: bool) -> float:
    """Helper function to compute y-coordinate (key rate) in case of
    isotropic noise scenario.

    @param x: Noise parameter (C/d/sec).
    @param d: The global dimension of input matrices.
    @param k: Dimension of subspaces within input matrices.
    @param f_matrix: Input Fourier matrix.
    @param c_matrix: Input computational matrix.
    @param is_bpsc: Switch between computing BPSC and BPS keyrate.
    @returns: Key rate (BPSC or BPS).
    """
    C = floor(x * d * 25.0)

    return perform_symmetric_simulation(C, d, k, f_matrix, c_matrix, is_bpsc=is_bpsc)


def transmission_loss_fun(dB: float, d: int, k: int, f_matrix: DATA_MATRIX, c_matrix: DATA_MATRIX,
                          is_bpsc: bool) -> float:
    """Helper function to compute y-coordinate (key rate) in case of
    transmission loss simulation.

    @param dB: Loss parameter (decibels).
    @param d: The global dimension of input matrices.
    @param k: Dimension of subspaces within input matrices.
    @param f_matrix: Input Fourier matrix.
    @param c_matrix: Input computational matrix.
    @param is_bpsc: Switch between computing BPSC and BPS keyrate.
    @returns: Key rate (BPSC or BPS)."""
    return perform_asymmetric_simulation(dB, d, k, f_matrix, c_matrix, is_bpsc=is_bpsc)


def isolated_source_fun(dB: float, d: int, k: int, f_matrix: DATA_MATRIX, c_matrix: DATA_MATRIX,
                        is_bpsc: bool, Sb: int = 0) -> float:
    """Helper function to compute y-coordinate (key rate) in case of
    isolated entanglement source scenario.

    @param dB: Loss parameter (decibels).
    @param d: The global dimension of input matrices.
    @param k: Dimension of subspaces within input matrices.
    @param f_matrix: Input Fourier matrix.
    @param c_matrix: Input computational matrix.
    @param is_bpsc: Switch between computing BPSC and BPS keyrate.
    @param Sb: Single count rate introduced to Bob as a noise.
    @returns: Key rate (BPSC or BPS)."""
    f_loss = compute_loss(dB, sum(f_matrix))
    c_loss = compute_loss(dB, sum(c_matrix))
    C_f = int(2 * f_loss * Sb * 5e-9 * 25.0)
    C_c = int(2 * c_loss * Sb * 5e-9 * 25.0)

    return perform_asymmetric_simulation(dB, d, k, f_matrix, c_matrix,
                                         is_bpsc=is_bpsc, isolated_source=True, C_f=C_f, C_c=C_c)


def network_fun(dB: float, k: int, f_matrix: DATA_MATRIX, c_matrix: DATA_MATRIX, F: int,
                is_bpsc: bool, index: int, subspace_index: int) -> float:
    """Helper function to compute y-coordinate (key rate) in case of
    simple QKD network scenario.

    @param dB: Lost parameter (decibels).
    @param k: Dimension of subspace currently processed.
    @param f_matrix: Input Fourier matrix.
    @param c_matrix: Input computational matrix.
    @param F: Channel noise introduced into Bob's lab.
    @param is_bpsc: Switch between computing BPSC and BPS keyrate.
    @param index: Cell index within input matrices, where currently
    processed subspace begins.
    @param subspace_index: Index of currently processed subspace.
    @returns: Key rate (BPSC or BPS)."""
    loss_comp = compute_loss(dB, sum(c_matrix))
    loss_four = compute_loss(dB, sum(f_matrix))

    modified_cmtrx = loss_matrix(c_matrix, dB)
    modified_fmtrx = loss_matrix(f_matrix, dB)

    C_comp = 2 * (loss_comp / (8 / k)) * F * 5e-9 * 25.0
    C_four = 2 * (loss_four / (8 / k)) * F * 5e-9 * 25.0

    modified_cmtrx = matrix_add(k, modified_cmtrx, noise_matrix(k, int(C_comp)), index, 8)
    modified_fmtrx = matrix_add(k, modified_fmtrx, noise_matrix(k, int(C_four)), index, 8)

    prob_cmtrx = [0.0 for _ in range(8 * 8)]
    prob_fmtrx = [0.0 for _ in range(8 * 8)]

    # if d8k224_subspaces_sum(modified_cmtrx) > 100 or d8k224_subspaces_sum(modified_fmtrx) > 100:
    prob_cmtrx = d8k224_probability_matrix(modified_cmtrx)
    prob_fmtrx = d8k224_probability_matrix(modified_fmtrx)

    keyrate = d8k224_keyrate_per_subspace(prob_cmtrx, prob_fmtrx)[subspace_index]

    if is_bpsc:
        return keyrate

    return keyrate * d8k224_subspaces_sum(modified_fmtrx) / 25


# Computes y-values for given x-values using given function. Y-values belongs to single line in specific graph.
def compute_y_coords(x_coords: List[float], function, *args) -> List[float]:
    """Computes y-coordinates (key rates) for each given x.

    @param x_coords: List of input x-coordinates.
    @param function: A function which outputs y-coordinate for each x.
    @param args: Arguments for given function.
    @returns: Y-coordinates.
    """

    y_coords = [0 for _ in x_coords]

    for i, x in enumerate(x_coords):
        y_coords[i] = function(x, *args)

    return y_coords


def make_graph(d_spaces: List[int], k_subspaces: List[int],
               f_matrices: List[DATA_MATRIX], c_matrices: List[DATA_MATRIX], function,
               start: int = 0, end: int = 10000, step: int = 100, divisor: int = 1,
               graph_title="Default Graph Title", xlabel="x axis", ylabel="y axis",
               is_bpsc=True, Sb: int = None, sci_notation: bool = True,
               printing: bool = False):
    """Creates a graph depicting the results of specific simulation.

    @param d_spaces: List of global dimensions for each QKD instance.
    @param k_subspaces: List of subspace dimensions for each QKD instance.
    @param f_matrices: List of input Fourier matrices.
    @param c_matrices: List of input computational matrices.
    @param function: Helper function for computing key rate.
    @param start: Lower bound of values passed to a function.
    @param end: Upper bound of values passed to a function.
    @param step: Interval of picking values between <start, end>.
    @param divisor: Value for dividing picked number from the interval.
    @param graph_title: Title of graph.
    @param xlabel: X-axis label.
    @param ylabel: Y-axis label.
    @param is_bpsc: Switch for drawing results of BPSC or BPS keyrate.
    @param Sb: Single count rate introduced to Bob's lab.
    @param sci_notation: Turn on/off scientific notation on axes.
    @param printing: Print informational messages into terminal.
    @returns: Graph object with drawn results.
    """
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']
    # x-coordinates for every QKD instance
    instances_xcoords = [[x / divisor for x in range(start, end + 1, step)] for _ in d_spaces]

    if sci_notation:
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)

    plt.xlabel(xlabel, fontsize=15)  # setting x-axis label
    plt.ylabel(ylabel, fontsize=15)  # setting y-axis label
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(graph_title)  # setting graph title

    if printing:
        print("Computing " + graph_title + "...")

    for instance_idx, x_coords in enumerate(instances_xcoords):
        d = d_spaces[instance_idx]
        k = k_subspaces[instance_idx]
        f_matrix = f_matrices[instance_idx]
        c_matrix = c_matrices[instance_idx]
        label = "d=" + str(d) + ",k=" + str(k)
        args = (d, k, f_matrix, c_matrix, is_bpsc)

        if Sb is not None:
            args = (*args, Sb)

        if printing:
            print("\td", d, "k", k, ": ", end="")

        y_coords = compute_y_coords(x_coords, function, *args)

        plt.plot(x_coords, y_coords, label=label, color=colors[instance_idx])

        if printing:
            print("FINISHED")

    plt.grid()
    plt.legend()
    plt.ylim(0, plt.ylim()[1])  # setting upper bound to y-axis according to actual max value on y-axis.

    return plt.gca()


def make_qkd_network_graph(f_matrix: DATA_MATRIX, c_matrix: DATA_MATRIX, F: int,
                           start: int = 0, step: int = 10, end: int = 100, divisor: int = 1,
                           graph_title="Title", xlabel="x axis", ylabel="y axis",
                           is_bpsc: bool = True, printing: bool = False):
    """Creates a graph depicting the results of \"Simple QKD network\"
    scenario.

    @param f_matrix: Input Fourier matrix.
    @param c_matrix: Input computational matrix.
    @param F: The amount of channel noise injected into Bob's lab.
    @param start: Lower bound of values passed to a function.
    @param end: Upper bound of values passed to a function.
    @param step: Interval of picking values between <start, end>.
    @param divisor: Value for dividing picked number from the interval.
    @param graph_title: Title of graph.
    @param xlabel: X-axis label.
    @param ylabel: Y-axis label.
    @param is_bpsc: Switch for drawing results of BPSC or BPS keyrate.
    @param printing: Print informational messages into terminal.
    @returns: Graph object with drawn results.
    """
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']
    subscripts = ['₁', '₂', '₃']
    x_coords = [x / divisor for x in range(start, end + 1, step)]
    subspace_sizes = [2, 2, 4]
    index = 0

    plt.xlabel(xlabel, fontsize=15)  # setting x-axis label
    plt.ylabel(ylabel, fontsize=15)  # setting y-axis label
    plt.title(graph_title)

    if printing:
        print("Computing " + graph_title + "...")

    for subspace_index in range(3):
        k = subspace_sizes[subspace_index]
        label = "Bob" + subscripts[subspace_index] + " k=" + str(k)

        if printing:
            print(f"\t{label}: ", end="")

        y_coords = compute_y_coords(x_coords, network_fun, k, f_matrix, c_matrix, F, is_bpsc, index, subspace_index)

        if printing:
            print("FINISHED")

        plt.plot(x_coords, y_coords, label=label, color=colors[subspace_index])
        index += 18

    plt.grid()
    plt.legend()
    plt.ylim(0, plt.ylim()[1])
    return plt.gca()


