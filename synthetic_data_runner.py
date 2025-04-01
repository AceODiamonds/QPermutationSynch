import os
import json
import numpy as np

import implementations.implementation3 as imp3

import utils.syn_data_processing as sdp
import matplotlib.pyplot as plt

import re
from dwave.samplers import SimulatedAnnealingSampler
import utils.data_processing as dp

# Step 1: Generate synthetic data
keypoints, abs_perm_dict, noisy_rel_perms = sdp.generate_synthetic_dataset(
    num_views=3,
    num_points=10,
    swap_ratio=0.2,
    noise_std=0.5,
    output_dir="synthetic_data"
)

# Step 2: Convert absolute permutation vectors to matrices

keypoints , abs_X , P = sdp.generate_synthetic_dataset(output_dir="synthetic_data")
# print(P)
# for k, v in P.items():
#     print(type(v))
P = {k: np.array(v) for k, v in P.items()}

model = imp3.qubo_formulation(P,3, penalty=1.5)
dwave_qubo = model.Q
res = SimulatedAnnealingSampler().sample_qubo(dwave_qubo)
best_sample = res.first.sample
matrices = imp3.dwave_sim_ann_absolute_permutation_extractor(best_sample)
if 'X1' not in matrices:
    n = next(iter(matrices.values())).shape[0]
    matrices['X1'] = np.eye(n, dtype=int)
num_trials = 30
X_true = {
'X1': np.eye(10, dtype=int),
'X2': P['P12'].T,
'X3': P['P13'].T,
}

print(matrices)
accuracies = []
for i in range(num_trials):
    res = SimulatedAnnealingSampler().sample_qubo(dwave_qubo)
    best_sample = res.first.sample
    matrices = imp3.dwave_sim_ann_absolute_permutation_extractor(best_sample)
    acc = imp3.accuracy_bits(matrices, X_true)
    accuracies.append(acc)
for name, X in matrices.items():
    print(f"{name} is valid:", imp3.is_valid_permutation_matrix(X))
print(f'Th min energy achieved through simulated annealing is : {res.first.energy}')
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f"Accuracy over {num_trials} trials: {mean_acc:.3f} ± {std_acc:.3f}")
P_12 = P['P12']
P_13= P['P13']
P_23 = P['P23']
X1 = np.eye(10,dtype = int)
# cycle consistency error
cycle_err = imp3.cycle_consistency_error(P, matrices)
print(f"Cycle consistency error: {cycle_err}")
# print(best_sample)
matrices = imp3.dwave_sim_ann_absolute_permutation_extractor(best_sample)
if 'X1' not in matrices:
    n = next(iter(matrices.values())).shape[0]
    matrices['X1'] = np.eye(n, dtype=int)

X_true = {
    'X1': np.eye(10, dtype=int),
    'X2': P['P12'].T,
    'X3': P['P13'].T,
    }
for name, X in matrices.items():
        print(f"{name} is valid:", imp3.is_valid_permutation_matrix(X))
print(f'Th min energy achieved through simulated annealing is : {res.first.energy}')
print(matrices)