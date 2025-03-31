import utils.data_processing as dp
from scipy.spatial.distance import cdist
import numpy as np
import re

from munkres import Munkres, print_matrix

import qubovert as qv
from qubovert import boolean_var, QUBO
from qubovert.sim import anneal_qubo

#extract the keypoints from the .mat files, done in data_procesing.py
# create permutation matrices
#use the keypoint extractor to get the keypoints and build the permutation matrices first

def pair_distance_calculator(folder_path1, folder_path2):
    coordinates1, coordinates2 = dp.pair_coordinates(folder_path1, folder_path2)
    distance_matrix = cdist(coordinates1, coordinates2, 'euclidean')
    return distance_matrix

#permutation matrix

def permutation_matrix(distance_matrix):
    n = len(distance_matrix)
    P = np.zeros((n,n), dtype = int)
    #use munkres(Hungarian) algorithm to find the optimal permutation
    mu = Munkres()
    indexes = mu.compute(distance_matrix)
    for row, column in indexes:
        P[row][column] = 1
    return P
# (noisy) permutation matrix is ready
#create the qubo formulation
#the goal is to find the optimal permutation matrix that minimizes the distance between the keypoints

#so I create two different set of variables


def qubo_formulation_2views(P): #only 2 images, so only one single pai-wise permutation matrix
    n = P.shape[0] # this will give the number of points on the images

    Xi = [[boolean_var(f'xi{i}{j}') for j in range(n)] for i in range(n)]
    Xj = [[boolean_var(f'xj{i}{j}') for j in range(n)] for i in range(n)]
    model = QUBO()
    #first Ferbius norm
    for k in range(n):
        for l in range(n):
            sum_m = sum(Xi[k][m]*Xj[l][m] for m in range(n))
            #model += (P[k][l] - sum_m)**2 ;;; expand it
            model += P[k][l] - 2*P[k][l]*sum_m + sum_m
    
    #now add constraints
    #row
    penalty = 1.5
    for k in range(n):
        row_Xi = sum(Xi[k])      # sum over row k of Xi
        row_Xj = sum(Xj[k])      # sum over row k of Xj
        model += penalty * (row_Xi - 1)**2
        model += penalty * (row_Xj - 1)**2
    #column
    for m in range(n):
        col_Xi = sum(Xi[k][m] for k in range(n))
        col_Xj = sum(Xj[k][m] for k in range(n))
        model += penalty * (col_Xi - 1)**2
        model += penalty * (col_Xj - 1)**2

    return model

#to extract the permutation matrices

def extract_absolute_permutations_2views(result_input):
    
    if isinstance(result_input, list):
        if len(result_input) == 0:
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
    
    pattern_xi = re.compile(r"xi(\d+)(\d+)")
    pattern_xj = re.compile(r"xj(\d+)(\d+)")
    
    xi_indices = []
    xj_indices = []

    # Collect indices for xi
    for key in result_state:
        if key.startswith("xi"):
            m = pattern_xi.match(key)
            if m:
                i = int(m.group(1))
                j = int(m.group(2))
                xi_indices.append((i, j))
                
    # Collect indices for xj
    for key in result_state:
        if key.startswith("xj"):
            m = pattern_xj.match(key)
            if m:
                i = int(m.group(1))
                j = int(m.group(2))
                xj_indices.append((i, j))
    
    if xi_indices:
        max_i = max(i for i, _ in xi_indices)
        max_j = max(j for _, j in xi_indices)
        n_rows = max_i + 1
        n_cols = max_j + 1
    else:
        return [], []
    
    Xi_matrix = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
    Xj_matrix = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
    
    # Fill in Xi matrix.
    for key, val in result_state.items():
        if key.startswith("xi"):
            m = pattern_xi.match(key)
            if m:
                i = int(m.group(1))
                j = int(m.group(2))
                Xi_matrix[i][j] = val
    
    # Fill in Xj matrix.
    for key, val in result_state.items():
        if key.startswith("xj"):
            m = pattern_xj.match(key)
            if m:
                i = int(m.group(1))
                j = int(m.group(2))
                Xj_matrix[i][j] = val
                
    return Xi_matrix, Xj_matrix


# if __name__ == "__main__":
#     data_path1 = r'./PF-dataset/car(G)/Cars_006a.mat'
#     data_path2 = r'./PF-dataset/car(G)/Cars_007a.mat'
#     distance_matrix = pair_distance_calculator(data_path1, data_path2)
#     #print(len(distance_matrix))
#     P = permutation_matrix(distance_matrix)
    
#     #the model
#     model = qubo_formulation_2views(P)
#     #solve with qubovert simulated annealing
#     result = anneal_qubo(model)
#     # for res in result:
#     #     print(res.value, res.state)
#     #extract the absolute_matrices
#     xi, xj = extract_absolute_permutations_2views(result)

#     print("Xi absolute permutation is: \n")
#     for row in xi:
#         print(row)
#     print("\nXj absolute permutation is: \n")
#     for row in xj:
#         print(row)

#     #now perform the multiplication of Xi and Xj^t
#     xi_nump = np.array(xi)
#     xj_nump = np.array(xj)
#     xj_nump_t = xj_nump.transpose()
#     #multiply
#     absolute_mult = np.matmul(xi_nump,xj_nump_t)
#     print("this is Pij: ")
#     print(P)
#     print("This is Xi@Xj^t:")
#     print(absolute_mult)
#     print("P - Xi*Xj^t:")
#     print(P - absolute_mult)