import os, sys
import utils.data_processing as dp
import utils.pair_methods as pm
import scipy
from datetime import datetime
import re
from collections import defaultdict, OrderedDict
#torch
import torch
from torchvision.models import alexnet
from torchvision import transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

#QUBOvert
import qubovert as qv
from qubovert import boolean_var, QUBO
from qubovert.sim import anneal_qubo, anneal_pubo
from dwave.samplers import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel



# step 0 : Handling folder paths

def path_joiner(image_name , root_dir=None): #recursive search for the image
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for dirpath , _, filenames in os.walk(root_dir):
        if image_name in filenames:
            return os.path.join(dirpath, image_name)


# step1 : load image keypoints

def keypoint(image_path):
    keypoints1 = scipy.io.loadmat(image_path)["pts_coord"]
    return keypoints1
#extract keypoints of several images
def keypoints_list(image_paths: list):
    keypoints = []
    for image_path in image_paths:
        keypoints1 = keypoint(image_path)
        keypoints.append(keypoints1)
    return keypoints

def keypoints_dict(image_names: list):
    keypoints = {}
    for image_name in image_names:
        base_name = os.path.splitext(image_name)[0]  # remove extension
        full_path = path_joiner(image_name)
        if full_path:
            keypoints[base_name] = keypoint(full_path)
        else:
            print(f"[Warning] image not found: {image_name}")
    return keypoints

# step2 : AlexNet and feature extraction
def alexnet_feature_extractor(layer= 'conv4'):
    model = alexnet(pretrained=True)
    model.eval()
    if layer == 'conv4':
        return torch.nn.Sequential(*list(model.features)[:10])
    elif layer == 'conv5':
        return torch.nn.Sequential(*list(model.features)[:12])
    else:
        raise ValueError("Invalid layer. Choose 'conv4' or 'conv5'.")

#extract features from keypoints
def extract_features(keypoints_dict, patch_size=64, layer='conv4'):
    feature_extractor = alexnet_feature_extractor(layer)
    transform = T.Compose([
        T.Resize((227, 227)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
    features = {}

    for image_name, keypoints in keypoints_dict.items():
        img_path = path_joiner(image_name + '.png')
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        feature_list = []

        for (x, y) in keypoints.T:
            x, y = int(round(x)), int(round(y))
            x1 = max(0, x - patch_size // 2)
            y1 = max(0, y - patch_size // 2)
            x2 = min(img.width, x + patch_size // 2)
            y2 = min(img.height, y + patch_size // 2)

            patch = img.crop((x1, y1, x2, y2))
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = feature_extractor(patch_tensor)
                feat = feat.mean(dim=[2, 3]) # to flatten the output dimensions
            feature_list.append(feat.squeeze().cpu().numpy())

        features[image_name] = np.stack(feature_list)

    return features
def visualize_similarity(features_dict, image_list):
    os.makedirs("data_collection", exist_ok=True)

    for i in range(len(image_list)):
        for j in range(i + 1, len(image_list)):
            img1 = image_list[i]
            img2 = image_list[j]
            feats1 = features_dict[img1]
            feats2 = features_dict[img2]
            sim_matrix = cosine_similarity(feats1, feats2)

            plt.figure(figsize=(6, 5))
            plt.imshow(sim_matrix, cmap='viridis')
            plt.colorbar(label='Cosine Similarity')
            plt.title(f"Similarity: {img1} vs {img2}")
            plt.xlabel(f"{img2} keypoints")
            plt.ylabel(f"{img1} keypoints")
            plt.tight_layout()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_collection/sim_{img1}_vs_{img2}_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
            print(f"[✓] Saved: {filename}")

def print_similarity_matrix(features_dict): # print the similarity(cost matrix)
    image_list = list(features_dict.keys())

    for i in range(len(image_list)):
        for j in range(i + 1, len(image_list)):
            img1 = image_list[i]
            img2 = image_list[j]

            feats1 = features_dict[img1]
            feats2 = features_dict[img2]
            sim_matrix = cosine_similarity(feats1, feats2)

            print(f"\n=== Cosine Similarity Matrix: {img1} vs {img2} ===")
            for row in sim_matrix:
                print("  ".join(f"{val:.2f}" for val in row))

def print_cost_matrix(features_dict): # this will be used for Hungarian algorithm
    image_list = list(features_dict.keys())

    for i in range(len(image_list)):
        for j in range(i + 1, len(image_list)):
            img1 = image_list[i]
            img2 = image_list[j]

            feats1 = features_dict[img1]
            feats2 = features_dict[img2]
            sim_matrix = cosine_similarity(feats1, feats2)
            cost_matrix = 1 - sim_matrix

            print(f"\n=== Cost Matrix (1 - cosine): {img1} vs {img2} ===")
            for row in cost_matrix:
                print("  ".join(f"{val:.2f}" for val in row))

def pairwise_permutations(features_dict): # compute the P_ij
    P = {}
    image_list = list(features_dict.keys())

    for i in range(len(image_list)):
        for j in range(i + 1, len(image_list)):
            img1 = image_list[i]
            img2 = image_list[j]

            feats1 = features_dict[img1]
            feats2 = features_dict[img2]
            similarity = cosine_similarity(feats1, feats2)
            cost_matrix = 1 - similarity

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            n = len(row_ind)
            perm_matrix = np.zeros((n, n), dtype=int)
            perm_matrix[row_ind, col_ind] = 1

            key = f"P{i+1}{j+1}"
            P[key] = perm_matrix

    return P

#QUBO formulation
def qubo_formulation(P: dict, num_views: int,set_identiy: int= 1, penalty=1.5):
    model = QUBO()

    # Get the number of keypoints from one relative permutation matrix
    valu = next(iter(P.values()))
    n = len(valu[0])

    # Create absolute permutation variables
    absolute_matrices = {}
    for i in range(num_views):
        if i == (set_identiy-1):
            absolute_matrices[f'X{i+1}'] = np.eye(n, dtype=int).tolist()
        else:
            absolute_matrices[f'X{i+1}'] = [
                [boolean_var(f'x{i+1}{k}{l}') for l in range(n)]
                for k in range(n)
            ]

    # === Objective term: matching error ===
    for i in range(num_views):
        for j in range(i + 1, num_views):
            key = f'P{i+1}{j+1}'
            P_ij = P[key]
            for k in range(n):
                for l in range(n):
                    abs_mult = sum(absolute_matrices[f'X{i+1}'][k][r] *
                                   absolute_matrices[f'X{j+1}'][l][r]
                                   for r in range(n))  # Xi * Xj^T
                    obj_term = (P_ij[k, l]) - 2 * (P_ij[k, l] * abs_mult) + abs_mult
                    model += obj_term

    # === Constraint: permutation structure ===
    for i in range(num_views):
        X = absolute_matrices[f'X{i+1}']
        for k in range(n):
            row_sum = sum(X[k][l] for l in range(n))
            model += penalty * (row_sum - 1) ** 2
        for l in range(n):
            col_sum = sum(X[k][l] for k in range(n))
            model += penalty * (col_sum - 1) ** 2

    # === Cycle Consistency ===
    # for i in range(num_views):
    #     for j in range(i + 1, num_views):
    #         for k in range(j + 1, num_views):
    #             Xi = absolute_matrices[f'X{i+1}']
    #             Xj = absolute_matrices[f'X{j+1}']
    #             Xk = absolute_matrices[f'X{k+1}']

    #             # Compute Xi * Xj^T * Xj * Xk^T vs. Xi * Xk^T
    #             for a in range(n):
    #                 for b in range(n):
    #                     lhs = sum(
    #                         Xi[a][r1] * Xj[b][r1] * sum(
    #                             Xj[b][r2] * Xk[b][r2]
    #                             for r2 in range(n)
    #                         ) for r1 in range(n)
    #                     )
    #                     rhs = sum(Xi[a][r] * Xk[b][r] for r in range(n))
    #                     cycle_term = lhs - 2 * lhs * rhs + rhs
    #                     model += cycle_penalty * cycle_term
    return model
#run simulated annealing based on the number passed as input

def run_dwave_simulated_annealing(model,num_trials: int = 1):
    sampler = SimulatedAnnealingSampler()
    Qubo_model = model.Q
    bqm = BinaryQuadraticModel.from_qubo(Qubo_model)
    res = sampler.sample(bqm, num_reads=num_trials)
    return res, res.first.sample, res.first.energy

def absolute_permutation_extractor(sample : dict, identity_view: int = 1):
    """
        Feed the best sample result of the annealing process to this function
        to extract the absolute matrices
    """
    pattern = re.compile(r"x(\d+)(\d+)(\d+)")
    view_vars = defaultdict(list)

    # group variables by view number
    for var, val in sample.items():
        match = pattern.match(var)
        if match:
            view = int(match.group(1))
            row = int(match.group(2))
            col = int(match.group(3))
            view_vars[view].append((row, col, val))

    # build matrices from ordered available view variables
    all_views = sorted(view_vars.keys())
    matrices = {}

    for view in all_views:
        entries = view_vars[view]
        # Figure out dimension
        n = max(max(r, c) for r, c, v in entries) + 1 if entries else 0
        matrix = np.zeros((n, n), dtype=int)
        for r, c, v in entries:
            matrix[r, c] = v

        matrices[f"X{view}"] = matrix

    # If the identity view is missing, add it as an identity matrix.
    if identity_view in all_views:
        # Try to determine dimension n from an existing matrix; 
        # if none exist, you might need to set a default value or raise an error.
        dim = matrices[f"X{identity_view}"].shape[0]
        matrices[f"X{identity_view}"] = np.eye(dim, dtype=int)
    else:
        
        if matrices:
            some_matrix = next(iter(matrices.values()))
            dim = some_matrix.shape[0]
        else:
            raise ValueError("No dimension could be inferred since sample is empty.")
        matrices[f"X{identity_view}"] = np.eye(dim, dtype=int)

    # return  them in sorted X1, X2, X3, ... order
    sorted_keys = sorted(matrices.keys(), key=lambda x: int(x[1:]))
    ordered_matrices = OrderedDict((k, matrices[k]) for k in sorted_keys)

    return ordered_matrices


def is_valid_permutation_matrix(X): #check the validity of the absolute permutation matrices(all rows and columns sum to 1)
    return (
        np.all(np.sum(X, axis=0) == 1) and
        np.all(np.sum(X, axis=1) == 1) and
        np.all((X == 0) | (X == 1))
    )
def cycle_consistency_error(P: dict, matrices: dict): # checks for cycle consistency error
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

#saving the final images and colored points across them
def save_visualization(
    image_paths,
    keypoints_list,
    permutation_matrices,
    X_true=None,
    energy=None,
    rel_perms=None,
    result_obj=None
):
    """
    Combines visualization of permutation matches (subplots of images and colored keypoints)
    with saving the final annotated image and text file of parameters.

    :param image_paths: List of image file paths (point .mat paths or similar),
                        though we load them as .png by replacement in this function.
    :param keypoints_list: List of keypoint arrays (each shape (N, 2) or (2, N)).
    :param permutation_matrices: Dict of absolute permutations, e.g. {'X1': np.eye(...), 'X2': ...}.
    :param X_true: (Optional) If you have any ground truth matrix you want to visualize or track.
    :param energy: (Optional) The final energy from simulated annealing or other solver.
    :param rel_perms: (Optional) Dictionary of relative permutations P_ij if you want them reported.
    :param result_obj: (Optional) Sampler or solver result object if you want to print sampler info.
    """

    # === 1) Construct a file prefix based on the object type in the first image name ===
    if 'duck' in image_paths[0].lower():
        prefix = 'd_'
    elif 'car' in image_paths[0].lower():
        prefix = 'c_'
    elif 'motorbike' in image_paths[0].lower():
        prefix = 'm_'
    else: #wine
        prefix = 'w_'

    # === 2) Build filename paths for saving ===
    image_codes = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    joined_name = "_".join(image_codes)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{prefix}{joined_name}_{timestamp}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    save_folder = os.path.join(parent_dir, 'synchronized_images_simulatedAnnealing')
    os.makedirs(save_folder, exist_ok=True)

    img_save_path = os.path.join(save_folder, base_filename + ".png")
    txt_save_path = os.path.join(save_folder, base_filename + ".txt")

    # === 3) Set up subplots for all views ===
    n_views = len(image_paths)
    fig, axes = plt.subplots(1, n_views, figsize=(5 * n_views, 5))
    if n_views == 1:
        axes = [axes]  # Ensure 'axes' is always iterable

    # === 4) Determine how many keypoints you have per image, define a color map ===
    num_points = len(keypoints_list[0])  # assumes each image has the same # of keypoints
    colors = plt.cm.get_cmap('tab10', num_points)

    # === 5) For each image, show it and scatter the points according to the permutation matrix ===
    for idx, (img_path, kp_array) in enumerate(zip(image_paths, keypoints_list)):
        # If your keypoints are shape (2, N), transpose to get (N, 2)
        if kp_array.shape[0] == 2 and kp_array.shape[1] != 2:
            kp_array = kp_array.T

        # Switch .mat extension to .png for display
        png_path = img_path.replace(".mat", ".png")
        img = mpimg.imread(png_path)
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')

        # Grab X{idx+1} or use identity if missing
        mat_key = f'X{idx+1}'
        if mat_key not in permutation_matrices:
            print(f"Warning: {mat_key} not found. Using identity matrix.")
            n = kp_array.shape[0]
            X = np.eye(n, dtype=int)
        else:
            X = permutation_matrices[mat_key]

        # For each original point index, find which row is "active" in that row of X
        # Then plot the corresponding keypoint:
        for point_idx in range(num_points):
            mapped_idx = np.argmax(X[point_idx])
            # x, y = kp_array[mapped_idx] -> if kp_array is shape (N, 2), each row is [x, y]
            x, y = kp_array[mapped_idx]
            ax.scatter(x, y, color=colors(point_idx), s=80, edgecolor='black')

    # === 6) Save the figure (the stitched images with annotation) ===
    plt.savefig(img_save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Saved visualization to {img_save_path}")

    # === 7) Save the .txt report about the run/solver ===
    with open(txt_save_path, "w") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Image paths: {', '.join(image_paths)}\n\n")

        if energy is not None:
            f.write(f"Simulated Annealing Final Energy: {energy:.6f}\n\n")

        if result_obj is not None and hasattr(result_obj, 'info'):
            f.write("Result Metadata (Sampler Info):\n")
            for k, v in result_obj.info.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        if rel_perms:
            f.write("Relative Permutation Matrices (P_ij):\n")
            for key in sorted(rel_perms.keys()):
                f.write(f"\n{key}:\n{rel_perms[key]}\n")

        f.write("\nEstimated Absolute Permutation Matrices (X_i):\n")
        for key in sorted(permutation_matrices.keys()):
            f.write(f"\n{key}:\n{permutation_matrices[key]}\n")

    print(f"[✓] Saved report to {txt_save_path}")



# run
image_names = ['Cars_006a.mat','Cars_007a.mat', 'Cars_008b.mat']
image_paths = []
for im in image_names:
    image_paths.append(path_joiner(im))
keypoints = keypoints_dict(image_names)
keypointslist = keypoints_list(image_paths=image_paths)
features = extract_features(keypoints, patch_size=64, layer='conv4')

# visualize similarities
visualize_similarity(features, list(features.keys()))
#permutation matrices
P = pairwise_permutations(features_dict=features)
#QUBO
model = qubo_formulation(P, 3, 1, 1.5)
#Sim_annealing
res, best_Sample, best_energy = run_dwave_simulated_annealing(model, 10)
#extract Xi....
abs_matrices = absolute_permutation_extractor(best_Sample) # it's a dictionary
#check validity
for _,Xi in abs_matrices.items():
    print(f'{Xi} is valid : {is_valid_permutation_matrix(Xi)}')
#check cycle consistency error
c_error = cycle_consistency_error(P, abs_matrices)
print(c_error)
print(f'The minimum energy achieved is : {best_energy}')
#save the visualizations
data_path1 = r'./PF-dataset/car(G)/Cars_006a.mat'
data_path2 = r'./PF-dataset/car(G)/Cars_007a.mat'
data_path3 = r'./PF-dataset/car(G)/Cars_008b.mat'
data_paths = [data_path1, data_path2, data_path3]

save_visualization(
        image_paths = data_paths, 
        keypoints_list= keypointslist, 
        permutation_matrices=abs_matrices, 
        X_true=None,
        energy=None, 
        rel_perms=None, 
        result_obj=None)
# print shape summary
for img, feats in features.items():
    print(f"{img} -> {feats.shape}")  # Should be (N, 256)

for k, v in P.items():
    print(f"{k} =\n{v}\n")


