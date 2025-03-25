import utils.data_processing as dp
import utils.pair_methods as pm
import numpy as np
import os
import re

#QUBOvert
import qubovert as qv
from qubovert import boolean_var, QUBO
from qubovert.sim import anneal_qubo
#######################################################################

def permutation_matrix_generator(*arg):
    num_views = len(arg)
    perm_mat = {}
    for i in range(num_views):
        for j in range(i+1, num_views):
            temp_dist = pm.pair_distance_calculator(arg[i], arg[j])
            perm_mat[f'P{i+1}{j+1}'] = pm.permutation_matrix(temp_dist)
    return perm_mat


def qubo_formulation(P: dict, num_views: int):
    model = QUBO()
    # get the number of keypoints from one relative permutation matrix
    valu = next(iter(P.values()))
    n = len(valu[0])
    
    absolute_matrices = {}
    for i in range(num_views):
        # Create an n x n absolute permutation matrix for view i.
        absolute_matrices[f'X{i}'] = [
            [boolean_var(f'x{i}{k}{l}') for l in range(n)]
            for k in range(n)
        ]
    
    # Objective: For each pair of views, we want to minimize the difference between
    # the measured relative permutation matrix and the product X_i * X_j^T.
    # (X_i * X_j^T)[k,l] = sum_r (X_i[k][r] * X_j[l][r]).
    for i in range(num_views):
        for j in range(i + 1, num_views):
            key = f'P{i+1}{j+1}'  # keys are 1-indexed in P
            P_ij = P[key]
            for k in range(n):
                for l in range(n):
                    rel_sum = sum(
                        absolute_matrices[f'X{i}'][k][r] * absolute_matrices[f'X{j}'][l][r]
                        for r in range(n)
                    )
                    # Under the permutation constraints, rel_sum is binary.
                    # Thus, (P_ij[k,l] - rel_sum)^2 = P_ij[k,l] - 2*P_ij[k,l]*rel_sum + rel_sum.
                    term = P_ij[k, l] - 2 * P_ij[k, l] * rel_sum + rel_sum
                    model += term

    # Add constraints to ensure each absolute permutation matrix is valid (each row and column sums to 1).
    penalty = 1000
    for i in range(num_views):
        X = absolute_matrices[f'X{i}']
        # Row constraints
        for k in range(n):
            row_sum = sum(X[k][l] for l in range(n))
            model += penalty * (row_sum - 1)**2
        # Column constraints
        for l in range(n):
            col_sum = sum(X[k][l] for k in range(n))
            model += penalty * (col_sum - 1)**2

    return model


#solving methods

#qubovert's simulated annealing

def simulated_annealing_qv(model):
    return anneal_qubo(model)
#extract the matrices

def extract_absolute_permutation_matrices(result_input):
    # Determine the state dictionary from the input.
    if isinstance(result_input, list):
        if not result_input:
            raise ValueError("The result list is empty.")
        if hasattr(result_input[0], "state"):
            result_state = result_input[0].state
        else:
            raise TypeError("Elements in the result list do not have a 'state' attribute.")
    elif hasattr(result_input, "state"):
        result_state = result_input.state
    elif isinstance(result_input, dict):
        result_state = result_input
    else:
        raise TypeError("Input must be an AnnealResult object, a list of AnnealResult objects, or a dictionary with variable states.")
    
    # Regular expression to match keys of the form "x{view}{row}{col}"
    # This works even if view, row, or col indices have more than one digit.
    pattern = re.compile(r"x(\d+)(\d+)(\d+)")
    
    # Group entries by view index.
    view_data = {}
    for key, val in result_state.items():
        m = pattern.match(key)
        if m:
            view_idx = int(m.group(1))
            row_idx = int(m.group(2))
            col_idx = int(m.group(3))
            view_data.setdefault(view_idx, []).append((row_idx, col_idx, val))
    
    # Build matrices for each view.
    absolute_matrices = {}
    for view_idx, entries in view_data.items():
        # Determine matrix dimensions.
        max_row = max(row for row, col, _ in entries)
        max_col = max(col for row, col, _ in entries)
        n_rows = max_row + 1
        n_cols = max_col + 1
        
        # Initialize matrix with zeros.
        mat = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
        # Fill matrix with values.
        for row, col, val in entries:
            mat[row][col] = val
        absolute_matrices[view_idx] = mat
    
    return absolute_matrices

#
def print_absolute_matrices(result_input):
    """
    Extracts and prints all absolute permutation matrices from the simulated annealing result.
    """
    absolute_matrices = extract_absolute_permutation_matrices(result_input)
    # Print each matrix, sorted by view index.
    for view_idx in sorted(absolute_matrices.keys()):
        print(f"Absolute matrix for view {view_idx}:")
        for row in absolute_matrices[view_idx]:
            print(row)
        print()  # Blank line between matrices











































#######################################################################
if __name__ == "__main__":

    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path1 = r'./PF-dataset/car(G)/Cars_006a.mat'
    data_path2 = r'./PF-dataset/car(G)/Cars_007a.mat'
    data_path3 = r'./PF-dataset/car(G)/Cars_008b.mat'
    #
    P = permutation_matrix_generator(data_path1,data_path2,data_path3)
    # print(P)
    m = qubo_formulation(P, 3)
    #
    #solve using qubovert simulated annealing
    result = anneal_qubo(m, num_anneals=1000)
    abs_mat = extract_absolute_permutation_matrices(result)

    x1 = abs_mat[0]
    x2 = abs_mat[1]
    x3 = abs_mat[2]
    #print the matrices
    # print_absolute_matrices(result)
    P12 = np.array(P['P12'])
    P13 = np.array(P['P13'])
    P23 = np.array(P['P23'])
    X1_nump = np.array(x1)
    X2_nump = np.array(x2)
    X3_nump = np.array(x3)
    print("this is X1")
    print(X1_nump)
    print("this is X2")
    print(X2_nump)
    print("this is X3")
    print(X3_nump)

    X1_nump_transpose = X1_nump.transpose()
    X2_nump_transpose = X2_nump.transpose()
    X3_nump_transpose = X3_nump.transpose()
    print("this is P12 and then P12 - X1X2")
    print(P12)
    print(P12 - np.matmul(X1_nump,X2_nump_transpose))
    print("this is P13 and then P13 - X1X3")
    print(P13)
    print(P13 - np.matmul(X1_nump,X3_nump_transpose))
    print("this is P23 and then P23 - X2X3")
    print(P23)
    print(P23 - np.matmul(X2_nump,X3_nump_transpose))