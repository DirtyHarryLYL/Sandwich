import pickle
import numpy as np

data = pickle.load(open("../Data/B123_train_KIN-FULL_with_node.pkl", "rb"))
cur_data = data[0] # read the 0-th sample
dataset_name, image_name, orig_action_labels, node_labels = cur_data
print("dataset_name", dataset_name)
print("image_name", image_name)
print("orig_action_labels", orig_action_labels)
print("node_labels", node_labels)

## mapping node to idx
mapping_node_index = pickle.load(open("../Data/mapping_node_index.pkl", "rb"))
verbnet_topology = pickle.load(open("../Data/verbnet_topology_898.pkl", "rb"))
Father2Son, objects = verbnet_topology["Father2Son"], verbnet_topology["objects"]
objects = np.array(objects)
objects_290 = objects[mapping_node_index]
for node_label in node_labels:
    print(node_label, np.where(objects_290==node_label)[0])