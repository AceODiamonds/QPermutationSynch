import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.spatial.distance import cdist


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








