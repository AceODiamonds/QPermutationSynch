import utils.data_processing as dp
import utils.pair_methods as pm
import numpy as np
import os
import re
from collections import defaultdict
#QUBOvert
import qubovert as qv
from qubovert import boolean_var, QUBO
from qubovert.sim import anneal_qubo, anneal_pubo
from dwave.samplers import SimulatedAnnealingSampler

#
from datetime import datetime
import matplotlib.pyplot as plt
#image processing
import scipy.io

#########################
'''
This implementation sets X1 to identity , 
'''
#########################

def permutation_matrix_generator(*arg):
    num_views = len(arg)
    perm_mat = {}
    for i in range(num_views):
        for j in range(i+1, num_views):
            temp_dist = pm.pair_distance_calculator(arg[i], arg[j])
            perm_mat[f'P{i+1}{j+1}'] = pm.permutation_matrix(temp_dist)
    return perm_mat

# using Gauge fixing ; 
def qubo_formulation(P: dict, num_views: int, penalty=1.5):
    model = QUBO()
    # get the number of keypoints from one relative permutation matrix
    valu = next(iter(P.values()))
    n = len(valu[0])
    
    absolute_matrices = {}
    for i in range(num_views):
        # Create an n x n absolute permutation matrix for view i.
        if i ==0:
            absolute_matrices[f'X{i+1}'] = np.eye(n, dtype=int).tolist() #X1 = Identity
        else:
            absolute_matrices[f'X{i+1}'] = [
                [boolean_var(f'x{i+1}{k}{l}') for l in range(n)]
                for k in range(n)
            ]
    #creating the Objective function
    for i in range(num_views):
        for j in range(i+1,num_views):
            key = f'P{i+1}{j+1}'
            P_ij = P[key]
            for k in range(n):
                for l in range(n):
                    abs_mult = sum(absolute_matrices[f'X{i+1}'][k][r] * 
                                   absolute_matrices[f'X{j+1}'][l][r]
                                   for r in range(n)
                                   ) # this is Xi*Xj^t
                    obj_term = (P_ij[k,l]) - 2*(P_ij[k,l]*abs_mult) + (abs_mult)
                    model += obj_term
    # add constraints
    for i in range(num_views):
        X = absolute_matrices[f'X{i+1}']
        #row constraint
        for k in range(n):
            row_sum = sum(X[k][l] for l in range(n))
            model += penalty * (row_sum - 1)**2
        # column constraint
        for l in range(n):
            col_sum = sum(X[k][l] for k in range(n))
            model += penalty * (col_sum - 1)**2
    return model
#solving
# result extraction

def qv_sim_ann_absolute_permutation_extractor(result_list):
    # Select the best (lowest-energy) result:
    best_result = result_list.best
    state_dict = best_result.state

    # Regular expression to capture view, row, and col from keys like "x200", "x310", etc.
    pattern = re.compile(r"x(\d+)(\d+)(\d+)")
    view_data = {}

    for key, val in state_dict.items():
        m = pattern.match(key)
        if m:
            view_idx = int(m.group(1))
            row_idx = int(m.group(2))
            col_idx = int(m.group(3))
            view_data.setdefault(view_idx, []).append((row_idx, col_idx, val))

    absolute_matrices = {}
    for view_idx, entries in view_data.items():
        max_row = max(r for r, c, v in entries)
        max_col = max(c for r, c, v in entries)
        # Initialize matrix with zeros
        mat = [[0 for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for r, c, v in entries:
            mat[r][c] = v
        absolute_matrices[view_idx] = np.array(mat)
    
    return absolute_matrices

#D-wave sampler absolute matrix extractor
def dwave_sim_ann_absolute_permutation_extractor(sample: dict):
    pattern = re.compile(r"x(\d+)(\d+)(\d+)")
    view_vars = defaultdict(list)

    # Group variables by view number
    for var, val in sample.items():
        match = pattern.match(var)
        if match:
            view = int(match.group(1))
            row = int(match.group(2))
            col = int(match.group(3))
            view_vars[view].append((row, col, val))

    # Build matrices
    matrices = {}
    for view, entries in view_vars.items():
        n = max(max(r, c) for r, c, v in entries) + 1
        matrix = np.zeros((n, n), dtype=int)
        for r, c, v in entries:
            matrix[r, c] = v
        matrices[f'X{view}'] = matrix

    return matrices
#evaluation and accuracy checking
def is_valid_permutation_matrix(X): #check the validity of the absolute permutation matrices(all rows and columns sum to 1)
    return (
        np.all(np.sum(X, axis=0) == 1) and
        np.all(np.sum(X, axis=1) == 1) and
        np.all((X == 0) | (X == 1))
    )
def accuracy_bits(X_est, X_true):
    total_bits = 0
    correct_bits = 0
    for key in X_est:
        est = X_est[key]
        true = X_true[key]
        correct_bits += np.sum(est == true)
        total_bits += est.size
    return correct_bits / total_bits
#cycle consistency error
def cycle_consistency_error(P: dict, matrices: dict):
    """
    Computes the total cycle consistency error over all triplets of views.
    P: dict of relative permutation matrices (keys like 'P12', 'P13', ...)
    matrices: dict of absolute permutation matrices (keys like 'X1', 'X2', ...)
    """
    views = sorted(int(k[1:]) for k in matrices.keys())  # Extract view numbers from 'X1', 'X2', ...
    total_error = 0
    count = 0

    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            for k in range(j + 1, len(views)):
                vi, vj, vk = views[i], views[j], views[k]
                key_ij = f'P{vi}{vj}'
                key_jk = f'P{vj}{vk}'
                key_ik = f'P{vi}{vk}'

                if key_ij in P and key_jk in P and key_ik in P:
                    lhs = P[key_ik]
                    rhs = P[key_ij] @ P[key_jk]
                    err = np.sum(np.abs(lhs - rhs))
                    total_error += err
                    count += 1

    avg_error = total_error / count if count > 0 else 0
    return avg_error




### main entry point
# if __name__ == "__main__":
#     parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     data_path1 = r'./PF-dataset/car(G)/Cars_006a.mat'
#     data_path2 = r'./PF-dataset/car(G)/Cars_007a.mat'
#     data_path3 = r'./PF-dataset/car(G)/Cars_008b.mat'
#     P = permutation_matrix_generator(data_path1,data_path2,data_path3)
#     model = qubo_formulation(P,3, penalty=1.5)
#     # solve using the D-Wave simulated annealing
#     dwave_qubo = model.Q
#     # solve with D-Wave
#     res = SimulatedAnnealingSampler().sample_qubo(dwave_qubo)
#     best_sample = res.first.sample
#     matrices = dwave_sim_ann_absolute_permutation_extractor(best_sample)
#     for name, X in matrices.items():
#         print(f"{name} is valid:", is_valid_permutation_matrix(X))
#     print(res.first.energy)
#     print(matrices)
#     # print(res.info)
#     #validate the accuracy
#     X_true = {
#     'X1': np.eye(10, dtype=int),
#     'X2': P['P12'].T,
#     'X3': P['P13'].T
#     }
#     num_trials = 20
#     accuracies = []

#     for i in range(num_trials):
#         res = SimulatedAnnealingSampler().sample_qubo(dwave_qubo)
#         best_sample = res.first.sample
#         matrices = dwave_sim_ann_absolute_permutation_extractor(best_sample)
#         acc = accuracy_bits(matrices, X_true)
#         accuracies.append(acc)

#     mean_acc = np.mean(accuracies)
#     std_acc = np.std(accuracies)

#     print(f"Accuracy over {num_trials} trials: {mean_acc:.3f} ± {std_acc:.3f}")
#     P_12 = P['P12']
#     P_13= P['P13']
#     P_23 = P['P23']
#     # X2 = matrices['X2']
#     # X3 = matrices['X3']
#     X1 = np.eye(10,dtype = int)
#     # X2_nump = np.array(X2)
#     # X3_nump = np.array(X3)
#     # #result checking
#     # print('P12 - X1X2T')
#     # print(P_12 - (X1@(X2_nump.transpose())))
#     # print('P13 - X1X3T')
#     # print(P_13 - (X1@(X3_nump.transpose())))
#     # print('P23 - X2X3T')
#     # print(P_23 - (X2@(X3_nump.transpose())))

#     # check for cycle consistencyP_12 = matrices['X1'] @ matrices['X2'].T
#     P_23 = matrices['X2'] @ matrices['X3'].T
#     P_13_est = X1 @ matrices['X3'].T

#     P_13_via_cycle = P_12 @ P_23

#     print("Cycle consistency error:", np.sum(np.abs(P_13_est - P_13_via_cycle))) # should result in a value close to 0

    