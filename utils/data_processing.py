import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from scipy.spatial.distance import cdist
from datetime import datetime
import json

#this script serves the prurpose of loading the image data
# that are in particular the willowObject dataset taken from the following link
# https://www.di.ens.fr/willow/research/proposalflow/

def load_image_data(category, image_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    dataset_path = os.path.join(parent_dir, 'PF-dataset')  # Use your actual folder name

    # Rest of your code remains the same
    image_path = os.path.join(dataset_path, f"{category}/", image_name)
    annotation_path = os.path.join(dataset_path, f"{category}/", image_name.replace(".png", ".mat"))

    #loading the image
    image = plt.imread(image_path)

    #loading the keypoints
    annotations = scipy.io.loadmat(annotation_path)
    keypoints = annotations["pts_coord"]

    

    #plotting the image and the keypoints
    plt.figure()
    plt.imshow(image)
    plt.scatter(keypoints[0, : ], keypoints[1, : ], c="red", marker = "x")

    plt.title(f"{category} - {image_name}")
    plt.show(block=False)
    plt.pause(0.1)
    plt.show()


# a method that goes through a folder and extracts the keypoints for each and stores them in a dictionary 
# with the name of the image as the key
def overall_keypoint_extractor(folder_path):
    keypoints = {}
    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            annotations = scipy.io.loadmat(os.path.join(folder_path, file))
            key,_ = os.path.splitext(file)
            keypoints[key] = annotations["pts_coord"]
    return keypoints


#start with something simple

def pair_keypoints(file1, file2):
    keypoints1 = scipy.io.loadmat(file1)["pts_coord"]
    keypoints2 = scipy.io.loadmat(file2)["pts_coord"]
    return keypoints1, keypoints2



#

def pair_coordinates(file1,file2):
    coordinates1 = [] # will contain all [x_i,y_i]
    coordinates2 = [] # will contain all [x_j,y_j]
    keypoints1 , keypoints2 = pair_keypoints(file1, file2)
    for i in range(len(keypoints1[0])):
        coordinate_i = np.array([keypoints1[0][i], keypoints1[1][i]])
        coordinate_j = np.array([keypoints2[0][i], keypoints2[1][i]])
        coordinates2.append(coordinate_j)
        coordinates1.append(coordinate_i)
    return coordinates1, coordinates2
#visualization and image post-processing

def get_all_keypoints(*file_paths):
    all_coords =[]
    for path in file_paths:
        coords1, _ = pair_coordinates(path,path)
        all_coords.append(coords1)
    return all_coords
def visualize_matches(image_paths, keypoints_list, permutation_matrices, X_true=None):
    n_views = len(image_paths)
    fig, axes = plt.subplots(1, n_views, figsize=(5*n_views, 5))
    if n_views == 1:
        axes=[axes]
    
    num_points = len(keypoints_list[0])

    #defining distinct colors
    colors = plt.cm.get_cmap('tab10' , num_points)
    # prepare mapping: X1 is identity, so X2 = P12.T etc.
    for idx, (img_path, keypoints) in enumerate(zip(image_paths, keypoints_list)):
        img = mpimg.imread(img_path.replace(".mat", ".png"))
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
        if f'X{idx+1}' not in permutation_matrices:
            print(f"Warning: X{idx+1} not found. Using identity matrix.")
            n = len(keypoints)
            X = np.eye(n, dtype=int)
        else:
            X = permutation_matrices[f'X{idx+1}']

        for point_idx in range(num_points):
            mapped_idx = np.argmax(X[point_idx])
            x, y = keypoints[mapped_idx]
            ax.scatter(x, y, color=colors(point_idx), s=80, edgecolor='black')

#save the image
def save_visualization(image_paths , keypoints_list, permutation_matrices, X_true = None , energy=None, rel_perms=None, result_obj=None):
    '''
        Saving the annotated final image and a .txt file
        containing important parameters and result of the annealing
    '''
    if 'duck' in image_paths[0].lower():
        prefix = 'd_'
    elif 'car' in image_paths[0].lower():
        prefix = 'c_'
    elif 'motorbike' in image_paths[0].lower():
        prefix = 'm_'
    else:
        prefix = 'w_'
    
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

    fig = plt.figure(figsize=(5 * len(image_paths), 5))
    visualize_matches(image_paths, keypoints_list, permutation_matrices, X_true=X_true)
    plt.savefig(img_save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Saved visualization to {img_save_path}")
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

#generate synthetic data
def generate_synthetic_permutation_data(
        num_agents = 5, num_items= 10, noise_level=0.2 , output_dir= "synthetic_data"
):
    os.makedirs(output_dir, exist_ok=True)

    #generate random permutation for each agent
    ground_truth_permutations = []
    for _ in range(num_agents):
        perm = np.random.permutation(num_items)
        ground_truth_permutations.append(perm.tolist())
    
    pairwise_permutations = {}
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                P_i = np.eye(num_items)[ground_truth_permutations[i]]
                P_j = np.eye(num_items)[ground_truth_permutations[j]]
                P_j_inv = np.linalg.inv(P_j)
                P_ij = P_i @ P_j_inv

                # Step 3: Add noise by swapping rows/columns
                num_swaps = int(noise_level * num_items)
                for _ in range(num_swaps):
                    a, b = np.random.choice(num_items, 2, replace=False)
                    P_ij[[a, b], :] = P_ij[[b, a], :]
                    P_ij[:, [a, b]] = P_ij[:, [b, a]]

                pairwise_permutations[f"{i}_{j}"] = P_ij.tolist()
    # Save results
    with open(os.path.join(output_dir, "ground_truth_permutations.json"), "w") as f:
        json.dump(ground_truth_permutations, f, indent=2)

    with open(os.path.join(output_dir, "pairwise_permutations.json"), "w") as f:
        json.dump(pairwise_permutations, f, indent=2)

    print(f"Synthetic data saved in '{output_dir}/'")
