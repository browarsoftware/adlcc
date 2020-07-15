
#https://scikit-learn.org/stable/modules/clustering.html
import pandas as pd
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn import metrics
import time

#https://scikit-learn.org/stable/modules/clustering.html

def compareAB(A, B, X):
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

    sc = metrics.silhouette_score(X, B, metric='euclidean')
    sc_str = '%17.3f' % sc
    chs = metrics.calinski_harabasz_score(X, B)
    chs_str = '%17.3f' % chs
    #dbs = metrics.davies_bouldin_score(X, B)
    #dbs_str = '%17.3f' % dbs

    my_str = ars_str + "&" + hs_str + "&" + cs_str + "&" + vms_str + "&" + fms_str + "&" + sc_str + "&" + chs_str #+ "&" + dbs_str
    return my_str

'''
identity = pd.read_csv("PCA_half/First_Half_faces.data.2020-06-18.csv")
all_data_to_cluster = pd.read_csv("PCA_half_res/AgglomerativeClustering_9_First_Half_faces.data.2020-06-18.csv.csv", header=None)
my_str = compareAB(identity.iloc[:, 2], all_data_to_cluster.iloc[:, 2])
print(my_str)
'''

#file_object = open('PCA_half_res/table.csv', 'w')
file_to_save_all = 'PCA_half_res/table.csv'

def append_to_file(file_path, string_to_save):
    file_object = open(file_path, 'a')
    file_object.write(string_to_save + '\n')
    file_object.close()



identity = pd.read_csv("PCA_half/Random_First_Half_same_twarze_faces_data_network_OK.txt")

file_name = "Random_First_Half_same_twarze_faces_data_network_OK.txt"

#for distance_threshold in [1, 2, 3]:#, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24]:
for distance_threshold in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]:
    strstr = 'PCA_half_res/AgglomerativeClustering' + '_' + str(distance_threshold) + '_' + file_name + '.csv'
    all_data_to_cluster = pd.read_csv(strstr, header=None)
    print(distance_threshold)



    my_str = compareAB(identity.iloc[:, 2], all_data_to_cluster.iloc[:, 2], identity.iloc[:, 3:150].to_numpy())
    output_str = strstr + "&" + my_str + "&" + str(len(pd.unique(all_data_to_cluster.iloc[:, 2])))
    #print(strstr + "&" + my_str)
    print(output_str)
    append_to_file(file_to_save_all, output_str)
    #file_object.write(output_str + '\n')

file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt"
for min_cluster_size in [2,3,4,5]:
    for number_of_pc in [21, 36, 48, 53, 62, 128]:
        strstr = 'PCA_half_res/PCA_' + str(number_of_pc) + '_hdbscan' + '_' + str(min_cluster_size) + '_' + file_name + '.csv'
        all_data_to_cluster = pd.read_csv(strstr, header=None)
        XXX = pd.read_csv("PCA_half/" + file_name)
        my_str = compareAB(identity.iloc[:, 2], all_data_to_cluster.iloc[:, 2],
                           XXX.iloc[:, 3:(3 + number_of_pc)].to_numpy())
        output_str = strstr + "&" + my_str + "&" + str(len(pd.unique(all_data_to_cluster.iloc[:, 2])))
        # print(strstr + "&" + my_str)
        print(output_str)
        append_to_file(file_to_save_all, output_str)
        #file_object.write(output_str + '\n')

file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt"
#file_name = "PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK.txt"
for distance_threshold in [10, 16]:
    for number_of_pc in [21, 36, 48, 53, 62, 128]:
        strstr = 'PCA_half_res/PCA_'+ str(number_of_pc) + '_AgglomerativeClustering' + '_' + str(distance_threshold) + '_' + file_name + '.csv'
        all_data_to_cluster = pd.read_csv(strstr, header=None)
        XXX = pd.read_csv("PCA_half/" + file_name)
        print(distance_threshold)
        my_str = compareAB(identity.iloc[:, 2], all_data_to_cluster.iloc[:, 2], XXX.iloc[:, 3:(3+number_of_pc)].to_numpy())
        output_str = strstr + "&" + my_str + "&" + str(len(pd.unique(all_data_to_cluster.iloc[:, 2])))
        #print(strstr + "&" + my_str)
        print(output_str)
        append_to_file(file_to_save_all, output_str)
        #file_object.write(output_str + '\n')


file_name = "Random_First_Half_same_twarze_faces_data_network_OK.txt"
for min_cluster_size in [2,3,4,5]:#range(10, 10):
    strstr = 'PCA_half_res/hdbscan' + '_' + str(min_cluster_size) + '_' + file_name + '.csv'
    print(strstr)
    all_data_to_cluster = pd.read_csv(strstr, header=None)
    my_str = compareAB(identity.iloc[:, 2], all_data_to_cluster.iloc[:, 2], identity.iloc[:, 3:150].to_numpy())
    output_str = strstr + "&" + my_str + "&" + str(len(pd.unique(all_data_to_cluster.iloc[:, 2])))
    # print(strstr + "&" + my_str)
    print(output_str)
    append_to_file(file_to_save_all, output_str)
    #file_object.write(output_str + '\n')
    

#for b in range(len(clustering.labels_)):
#    file_object.write(str(all_data_to_cluster.iloc[b, 1]) + ',' + str(all_data_to_cluster.iloc[b, 2]) + ',' + str(clustering.labels_[b]) + '\n')
#file_object.close()

