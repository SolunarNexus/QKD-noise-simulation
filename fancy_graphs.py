import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matrices import *
from simulator import *
import json


def qkd_net_write_json_simulation_runs(start: int, end: int, step: int, y_function, runs: int = 100,
                               json_filename: str = "stats.json", noise_amount: int = 0, is_qkd_network: bool = False):
    x_coords = [x / 100 for x in range(start, end + 1, step)]
    y_coords = [[] for _ in range(runs)]
    network_subdimensions = [2, 2, 4]

    print("QKD network simulation [F=" + str(noise_amount) + "]")

    for i in range(runs):
        print("Run " + str(i + 1) + ": ")
        cell_index = 0

        for setting_idx in range(3):
            k = network_subdimensions[setting_idx]
            # f_matrix = fourier_matrices[setting_idx]
            # c_matrix = computational_matrices[setting_idx]

            args = (k, d8k224_f_matrix, d8k224_c_matrix, noise_amount, True, cell_index, setting_idx)
            cell_index += 18
            print("\td=" + str(8) + ", k=" + str(k) + ": ", end="")
            y_coords[i].append(compute_y_coords(x_coords, y_function, *args))
            print("FINISHED")

    with open(json_filename, "w") as file:
        json.dump(y_coords, file, indent=4)


def write_json_simulation_runs(start: int, end: int, step: int, y_function, runs: int = 100,
                               json_filename: str = "stats.json"):
    x_coords = [x/100 for x in range(start, end + 1, step)]
    y_coords = [[] for _ in range(runs)]

    print("QKD simulation of Isolated source scenario...")

    for i in range(runs):
        print("Run " + str(i + 1) + ": ")

        for setting_idx, d in enumerate(global_dimensions):
            k = subspace_dimensions[setting_idx]
            f_matrix = fourier_matrices[setting_idx]
            c_matrix = computational_matrices[setting_idx]

            args = (d, k, f_matrix, c_matrix, True)
            print("\td=" + str(d) + ", k=" + str(k) + ": ", end="")
            y_coords[i].append(compute_y_coords(x_coords, y_function, *args))
            print("FINISHED")

    with open(json_filename, "w") as file:
        json.dump(y_coords, file, indent=4)


def get_bounding_lines(runs, instance_i):
    upper = [None] * 151
    lower = [None] * 151
    avg = [0] * 151

    for run in runs:
        for y_i in range(len(run[instance_i])):
            avg[y_i] += run[instance_i][y_i] / 100

            if upper[y_i] is None or upper[y_i] < run[instance_i][y_i]:
                upper[y_i] = run[instance_i][y_i]

            if lower[y_i] is None or lower[y_i] > run[instance_i][y_i]:
                lower[y_i] = run[instance_i][y_i]

    return upper, lower, avg


def create_fancy_graph(json_statsfile: str, xlabel: str, ylabel: str, settings_amount: int, labels: List[str] = None):
    file = open(json_statsfile, "r")
    x_coords = [x/100 for x in range(0, 3001, 20)]
    y_coords = json.load(file)

    file.close()
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']

    for i in range(settings_amount):
        upper, lower, avg = get_bounding_lines(y_coords, i)
        upper_smooth = savgol_filter(upper, 31, 1)
        lower_smooth = savgol_filter(lower, 31, 1)
        avg_smooth = savgol_filter(avg, 31, 1)

        if labels:
            label = labels[i]
        else:
            label = "d=" + str(global_dimensions[i]) + ",k=" + str(subspace_dimensions[i])

        plt.fill_between(x_coords, upper_smooth, lower_smooth, alpha=0.3, color=colors[i])
        plt.plot(x_coords, avg_smooth, color=colors[i], label=label)

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.ylim(0, plt.ylim()[1])
    plt.grid()
    plt.legend()

# make_qkd_network_graph(d8k224_f_matrix, d8k224_c_matrix, F=0, start=0, step=20, end=3000, divisor=100, printing=True)
# plt.show()

# qkd_net_write_json_simulation_runs(0, 3000, 20, network_fun, 100, "qkd_network_F0_100runs_extended30.json", 0, True)

# make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices, isolated_source_fun,
#            0, 3000, 20, 100, "", "xaxis", "yaxis", Sb=0, printing=True)
# plt.show()

create_fancy_graph("qkd_network_F0_100runs_extended30.json", xlabel="Decibels [dB]", ylabel="Key rate [BPSC]",
                   settings_amount=3, labels=["Bob₁ k=2", "Bob₂ k=2", "Bob₃ k=4"])
plt.savefig("qkd_network_F0_100runs_extended3000.pdf")
plt.show()

# write_json_simulation_runs(0, 3000, 20, isolated_source_fun, 100, "isolated_source_Sb0_100runs_extended30.json")

# create_fancy_graph("isolated_source_Sb0_100runs_extended30.json", xlabel="Decibels [dB]", ylabel="Key rate [BPSC]", settings_amount=5)
# plt.savefig("Isolated_source_Sb0_100runs_extended30.pdf")
# plt.show()
