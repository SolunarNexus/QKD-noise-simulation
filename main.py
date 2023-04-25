#!/usr/bin/env python

"""
This script is used for generating graphs depicting results
of our simulations as the main outcome of thesis.

This module is a part of the
"Examining noise models for high-dimensional Quantum
Key Distribution with subspace encoding" thesis.

@author Oskar Adam Valent
"""
import matplotlib.pyplot as plt
from simulator import \
    make_graph, make_qkd_network_graph, \
    marker_xcoord_detector_noise, \
    marker_xcoord_channel_noise, \
    marker_xcoord_isotropic_noise, \
    put_markers, \
    detector_noise_fun, channel_noise_fun, isotropic_noise_fun, isolated_source_fun
from matrices import fourier_matrices, computational_matrices, \
    d8k224_f_matrix, d8k224_c_matrix, \
    global_dimensions, subspace_dimensions, \
    avg_keyrates_bpsc, avg_keyrates_bps, \
    extra_coincidences

det_bpsc = make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
                      detector_noise_fun,
                      start=0, end=80000, step=100, graph_title="",
                      xlabel="S[s\u207b\u00b9]",
                      ylabel="Key rate [BPSC]", is_bpsc=True)
put_markers(det_bpsc, marker_xcoord_detector_noise, global_dimensions, subspace_dimensions, extra_coincidences,
            avg_keyrates_bpsc)
plt.savefig("Detector noise - BPSC.pdf")
plt.show()

det_bps = make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
                     detector_noise_fun,
                     start=0, end=80000, step=100, graph_title="", xlabel="S[s\u207b\u00b9]",
                     ylabel="Key rate [BPS]", is_bpsc=False)
put_markers(det_bps, marker_xcoord_detector_noise, global_dimensions, subspace_dimensions, extra_coincidences,
            avg_keyrates_bps)
plt.savefig("Detector noise - BPS.pdf")
plt.show()

chnl_bpsc = make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
                       channel_noise_fun,
                       start=0, end=600000, step=1000, graph_title="",
                       xlabel="d*S[s\u207b\u00b9]", ylabel="Key rate [BPSC]", is_bpsc=True)
put_markers(chnl_bpsc, marker_xcoord_channel_noise, global_dimensions, subspace_dimensions, extra_coincidences,
            avg_keyrates_bpsc)
plt.savefig("Channel noise - BPSC.pdf")
plt.show()

chnl_bps = make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
                      channel_noise_fun,
                      start=0, end=600000, step=1000, graph_title="",
                      xlabel="d*S[s\u207b\u00b9]", ylabel="Key rate [BPS]", is_bpsc=False, )
put_markers(chnl_bps, marker_xcoord_channel_noise, global_dimensions, subspace_dimensions, extra_coincidences,
            avg_keyrates_bps)
plt.savefig("Channel noise - BPS.pdf")
plt.show()

iso_bpsc = make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
                      isotropic_noise_fun,
                      start=0, end=400, step=2, graph_title="",
                      xlabel="Extra coincidences [s\u207b\u00b9d\u207b\u00b9]",
                      ylabel="Key rate [BPSC]", is_bpsc=True)
put_markers(iso_bpsc, marker_xcoord_isotropic_noise, global_dimensions, subspace_dimensions, extra_coincidences,
            avg_keyrates_bpsc)
plt.savefig("Isotropic noise - BPSC.pdf")
plt.show()

iso_bps = make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
                     isotropic_noise_fun,
                     start=0, end=400, step=2, graph_title="",
                     xlabel="Extra coincidences [s\u207b\u00b9d\u207b\u00b9]",
                     ylabel="Key rate [BPS]", is_bpsc=False)
put_markers(iso_bps, marker_xcoord_isotropic_noise, global_dimensions, subspace_dimensions, extra_coincidences,
            avg_keyrates_bps)
plt.savefig("Isotropic noise - BPS.pdf")
plt.show()

make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
           isolated_source_fun,
           start=0, end=2000, step=10, divisor=100,
           graph_title="",
           xlabel="Decibels [dB]", ylabel="Key rate [BPSC]",
           is_bpsc=True, Sb=0, sci_notation=False)
plt.savefig("Isolated source Sb=0 - BPSC.pdf")
plt.show()

make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
           isolated_source_fun,
           start=0, end=2000, step=10, divisor=100, graph_title="",
           xlabel="Decibels [dB]", ylabel="Key rate [BPSC]",
           is_bpsc=True, Sb=160000, sci_notation=False)
plt.savefig("Isolated source Sb=160k - BPSC.pdf")
plt.show()

make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
           isolated_source_fun,
           start=0, end=2000, step=10, divisor=100, graph_title="",
           xlabel="Decibels [dB]", ylabel="Key rate [BPSC]",
           is_bpsc=True, Sb=320000, sci_notation=False)
plt.savefig("Isolated source Sb=320k - BPSC.pdf")
plt.show()

make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
           isolated_source_fun,
           start=0, end=2000, step=10, divisor=100, graph_title="",
           xlabel="Decibels [dB]", ylabel="Key rate [BPSC]",
           is_bpsc=True, Sb=500000, sci_notation=False)
plt.savefig("Isolated source Sb=500k - BPSC.pdf")
plt.show()

make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
           isolated_source_fun,
           start=0, end=2000, step=10, divisor=100, graph_title="",
           xlabel="Decibels [dB]", ylabel="Key rate [BPS]",
           is_bpsc=False, Sb=0, sci_notation=False)
plt.savefig("Isolated source Sb=0 - BPS.pdf")
plt.show()

make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
           isolated_source_fun,
           start=0, end=2000, step=10, divisor=100, graph_title="",
           xlabel="Decibels [dB]", ylabel="Key rate [BPS]",
           is_bpsc=False, Sb=160000, sci_notation=False)
plt.savefig("Isolated source Sb=160k - BPS.pdf")
plt.show()

make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
           isolated_source_fun,
           start=0, end=2000, step=10, divisor=100, graph_title="",
           xlabel="Decibels [dB]", ylabel="Key rate [BPS]",
           is_bpsc=False, Sb=320000, sci_notation=False)
plt.savefig("Isolated source Sb=320k - BPS.pdf")
plt.show()

make_graph(global_dimensions, subspace_dimensions, fourier_matrices, computational_matrices,
           isolated_source_fun,
           start=0, end=2000, step=10, divisor=100, graph_title="",
           xlabel="Decibels [dB]", ylabel="Key rate [BPS]",
           is_bpsc=False, Sb=500000, sci_notation=False)
plt.savefig("Isolated source Sb=500k - BPS.pdf")
plt.show()

make_qkd_network_graph(d8k224_f_matrix, d8k224_c_matrix, F=0,
                       start=0, end=2000, step=10, divisor=100,
                       graph_title="", xlabel="Decibels [dB]", ylabel="Key rate [BPSC]",
                       is_bpsc=True)
plt.savefig("QKD network F=0 - BPSC.pdf")
plt.show()

make_qkd_network_graph(d8k224_f_matrix, d8k224_c_matrix, F=160000,
                       start=0, end=2000, step=10, divisor=100,
                       graph_title="", xlabel="Decibels [dB]", ylabel="Key rate [BPSC]",
                       is_bpsc=True)
plt.savefig("QKD network F=160k - BPSC.pdf")
plt.show()

make_qkd_network_graph(d8k224_f_matrix, d8k224_c_matrix, F=320000,
                       start=0, end=2000, step=10, divisor=100,
                       graph_title="", xlabel="Decibels [dB]", ylabel="Key rate [BPSC]",
                       is_bpsc=True)
plt.savefig("QKD network F=320k - BPSC.pdf")
plt.show()

make_qkd_network_graph(d8k224_f_matrix, d8k224_c_matrix, F=500000,
                       start=0, end=2000, step=10, divisor=100,
                       graph_title="", xlabel="Decibels [dB]", ylabel="Key rate [BPSC]",
                       is_bpsc=True)
plt.savefig("QKD network F=500k - BPSC.pdf")
plt.show()

make_qkd_network_graph(d8k224_f_matrix, d8k224_c_matrix, F=0,
                       start=0, end=2000, step=10, divisor=100,
                       graph_title="", xlabel="Decibels [dB]", ylabel="Key rate [BPS]",
                       is_bpsc=False)
plt.savefig("QKD network F=0 - BPS.pdf")
plt.show()

make_qkd_network_graph(d8k224_f_matrix, d8k224_c_matrix, F=160000,
                       start=0, end=2000, step=10, divisor=100,
                       graph_title="", xlabel="Decibels [dB]", ylabel="Key rate [BPS]",
                       is_bpsc=False)
plt.savefig("QKD network F=160k - BPS.pdf")
plt.show()

make_qkd_network_graph(d8k224_f_matrix, d8k224_c_matrix, F=320000,
                       start=0, end=2000, step=10, divisor=100,
                       graph_title="", xlabel="Decibels [dB]", ylabel="Key rate [BPS]",
                       is_bpsc=False)
plt.savefig("QKD network F=320k - BPS.pdf")
plt.show()

make_qkd_network_graph(d8k224_f_matrix, d8k224_c_matrix, F=500000,
                       start=0, end=2000, step=10, divisor=100,
                       graph_title="", xlabel="Decibels [dB]", ylabel="Key rate [BPS]",
                       is_bpsc=False)
plt.savefig("QKD network F=500k - BPS.pdf")
plt.show()
