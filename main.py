import implementations.implementation3 as imp3
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from dwave.samplers import SimulatedAnnealingSampler
import utils.data_processing as dp







if __name__ == "__main__":
    # run with 4 images
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path1 = r'./PF-dataset/car(S)/Cars_000a.mat'
    data_path2 = r'./PF-dataset/car(S)/Cars_001b.mat'
    data_path3 = r'./PF-dataset/car(S)/Cars_004b.mat'
    data_path4= r'./PF-dataset/car(S)/Cars_008a.mat'
    # data_path4 = r'./PF-dataset/duck(S)/060_0071.mat'
    # data_path4 = r'./PF-dataset/car(G)/.mat'
    # data_path5 = r'./PF-dataset/car(G)/Cars_017b.mat'
    # data_path6 = r'./PF-dataset/car(G)/Cars_019a.mat'
    P = imp3.permutation_matrix_generator(data_path1,data_path2,data_path3, data_path4)
    model = imp3.qubo_formulation(P,3, penalty=1.5)
    dwave_qubo = model.Q
    res = SimulatedAnnealingSampler().sample_qubo(dwave_qubo)
    best_sample = res.first.sample
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

    num_trials = 30
    accuracies = []
    for i in range(num_trials):
        res = SimulatedAnnealingSampler().sample_qubo(dwave_qubo)
        best_sample = res.first.sample
        matrices = imp3.dwave_sim_ann_absolute_permutation_extractor(best_sample)
        acc = imp3.accuracy_bits(matrices, X_true)
        accuracies.append(acc)
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
    #visualization
    image_paths = [data_path1, data_path2, data_path3,data_path4]
    keypoints_all = dp.get_all_keypoints(*image_paths)
    #saving the images and the parameters
    dp.save_visualization(
        image_paths=image_paths,
        keypoints_list=keypoints_all,
        permutation_matrices=matrices,
        X_true=X_true,
        energy=res.first.energy,
        rel_perms=P,
        result_obj=res.first
    )
