from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn import metrics

def compareAB(A, B):
    #measures the similarity of the two assignments, ignoring permutations and with chance normalization
    ars = metrics.adjusted_rand_score(A, B)
    ars_str = '%17.3f' % ars

    # each cluster contains only members of a single class
    hs = homogeneity_score(A, B)
    hs_str = '%17.3f' % hs

    #all members of a given class are assigned to the same cluster
    cs = completeness_score(A, B)
    cs_str = '%17.3f' % cs


    vms = metrics.v_measure_score(A, B)
    vms_str = '%17.3f' % vms


    # geometric mean of the pairwise precision and recall
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(A, B)
    fms_str = '%17.3f' % fowlkes_mallows_score


    my_str = ars_str + "&" + hs_str + "&" + cs_str + "&" + vms_str + "&" + fms_str
    return my_str

def append_to_file(file_path, string_to_save):
    file_object = open(file_path, 'a')
    file_object.write(string_to_save + '\n')
    file_object.close()

import pandas as pd

'''
n_neighbors = 1
for distance_threshold in [34]:
    #distance_threshold = 16
    train_file_name = "Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"
    valid_file_name = "Random_Second_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"
    cluster_file_name = "AgglomerativeClustering_" + str(distance_threshold) +  "_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt.csv"
    train_data = pd.read_csv('PCA_half/' + train_file_name)
    valid_data = pd.read_csv('PCA_half/' + valid_file_name)
    clustering_data = pd.read_csv('PCA_half_res/' + cluster_file_name, header=None)
    class_id = clustering_data.iloc[:, 2].to_numpy()
    X = train_data.iloc[:, 3:150].to_numpy()
    X_valid = valid_data.iloc[:, 3:150].to_numpy()
    id_true_valid = valid_data.iloc[:, 2].to_numpy()
    #print(class_id)
    #print(X)
    #print(X_valid)
    #print(id_true_valid)
    from sklearn.neighbors import KNeighborsClassifier
    for n_neighbors in [1]:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        neigh.fit(X, class_id)
        file_name = 'knn=' + str(n_neighbors) + "_" + train_file_name + "_" + cluster_file_name
        for index_help in range(id_true_valid.shape[0]):
            xxx = X_valid[index_help:(index_help + 1), ]
            pred_id = neigh.predict(xxx)[0]
            strstr = str(valid_data.iloc[index_help, 0]) + "," + str(id_true_valid[index_help]) + "," + str(pred_id)
            print(strstr)
            append_to_file('PCA_half_class_res/' + file_name, strstr)
'''

distance_threshold = 34
#number_of_pc = 21
train_file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"
valid_file_name = "PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"



train_data = pd.read_csv('PCA_half/' + train_file_name)
valid_data = pd.read_csv('PCA_half/' + valid_file_name)

id_true_valid = valid_data.iloc[:, 2].to_numpy()

from sklearn.neighbors import KNeighborsClassifier

for n_neighbors in [1]:
    for number_of_pc in [36, 48, 53, 62]:
        cluster_file_name = 'PCA_' + str(number_of_pc) + '_AgglomerativeClustering' + '_' + str(
            distance_threshold) + '_' + train_file_name + '.csv'
        print(cluster_file_name)
        clustering_data = pd.read_csv('PCA_half_res/' + cluster_file_name, header=None)
        class_id = clustering_data.iloc[:, 2].to_numpy()

        X = train_data.iloc[:, 3:150]
        X_valid = valid_data.iloc[:, 3:150]

        X = X.iloc[:, 0:number_of_pc].to_numpy()
        X_valid = X_valid.iloc[:, 0:number_of_pc].to_numpy()

        #print(X.shape)
        #print(X_valid.shape)

        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        neigh.fit(X, class_id)
        file_name = 'knn=' + str(n_neighbors) + "_" + train_file_name + "_" + cluster_file_name
        '''
        index_help = 0
        xxx = X_valid[index_help:(1+index_help), ]
        pred_id = neigh.predict(xxx)
        print(pred_id)
        '''

        for index_help in range(id_true_valid.shape[0]):
            xxx = X_valid[index_help:(index_help + 1), ]
            pred_id = neigh.predict(xxx)[0]

            strstr = str(valid_data.iloc[index_help, 0]) + "," + str(id_true_valid[index_help]) + "," + str(pred_id)
            #print(strstr)
            if (index_help % 1000 == 0):
                print(index_help)
            append_to_file('PCA_half_class_res/' + file_name, strstr)