import utils.data_processing as dp
import utils.pair_methods as pm
import numpy as np
import os
import re
#QUBOvert
import qubovert as qv
from qubovert import boolean_var, QUBO
from qubovert.sim import anneal_qubo, anneal_pubo

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

def extract_best_absolute_matrices(result_list):
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


if __name__ == "__main__":
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path1 = r'./PF-dataset/car(G)/Cars_006a.mat'
    data_path2 = r'./PF-dataset/car(G)/Cars_007a.mat'
    data_path3 = r'./PF-dataset/car(G)/Cars_008b.mat'
    P = permutation_matrix_generator(data_path1,data_path2,data_path3)
    model = qubo_formulation(P,3, penalty=1.5)
    result = anneal_qubo(model)
    best_matrices = extract_best_absolute_matrices(result)
    #For example, print the matrices for view 1, view 2, view 3:
    # for view in sorted(best_matrices.keys()):
    #     print(f"Absolute matrix for view {view}:")
    #     print(best_matrices[view])
    #     print()
    X3_nump_t = np.array(best_matrices[3]).transpose()
    X1 =  np.eye(10, dtype=int)
    print("P13 AND X1@X3T::")
    print(np.array(P['P12']))
    print(X1)
    print("this is X3")
    print(best_matrices[3])
    print(X3_nump_t)
    print(np.matmul(X1,X3_nump_t))
    print(np.array(P['P12']) - (np.matmul(X1,X3_nump_t)))
    
    # print(np.array(P['P12']))