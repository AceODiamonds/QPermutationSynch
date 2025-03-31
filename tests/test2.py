import utils.data_processing as dp
import os




parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path1 = os.path.join(parent_directory, "PF-dataset/car(G)/Cars_006a.mat")
file_path2 = os.path.join(parent_directory, "PF-dataset/car(G)/Cars_007a.mat")
file_path3 = os.path.join(parent_directory, "PF-dataset/car(G)/Cars_008b.mat")
file_path4 = os.path.join(parent_directory, "PF-dataset/car(G)/Cars_009b.mat")


#
coordinates1,coordinates2 = dp.pair_coordinates(file_path1, file_path2)



# keypoints1, keypoints2 = pair_keypoints(file_path1, file_path2)
# print(len(keypoints1))
# print(keypoints2)