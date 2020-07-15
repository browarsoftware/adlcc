from sklearn.cluster import AgglomerativeClustering
import numpy as np
'''
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5).fit(X)
print(clustering)
AgglomerativeClustering()
print(clustering.labels_)

'''
# https://scikit-learn.org/stable/modules/clustering.html
import pandas as pd
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn import metrics
import time


def compareAB(A, B):
    # measures the similarity of the two assignments, ignoring permutations and with chance normalization
    ars = metrics.adjusted_rand_score(A, B)
    print("adjusted_rand_score " + str(ars))
    # measures the agreement of the two assignments, ignoring permutations
    # amis = metrics.adjusted_mutual_info_score(A, B)
    # print("adjusted_mutual_info_score " + str(amis))
    # each cluster contains only members of a single class
    hs = homogeneity_score(A, B)
    print("homogeneity_score " + str(hs))
    # all members of a given class are assigned to the same cluster
    cs = completeness_score(A, B)
    print("completeness_score " + str(cs))
    vms = metrics.v_measure_score(A, B)
    print("v_measure_score " + str(vms))
    # geometric mean of the pairwise precision and recall
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(A, B)
    print("fowlkes_mallows_score " + str(fowlkes_mallows_score))

import pandas as pd
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn import metrics
import time

#file_name = "30-16faces.data.attr.csv"
#file_name = "faces.data.2020-06-18.csv"
#all_data_to_cluster_names = pd.read_csv(file_name)
identity = pd.read_csv("PCA_half/PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt")

file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt"
all_data_to_cluster = pd.read_csv("PCA_half/" + file_name).iloc[:, 3:150]
start_time = time.time()
#clusterin_labels = clusterer.fit_predict(all_data_to_cluster.iloc[:, 2:130])
distance_threshold = 16
#my_range = 100000
print(all_data_to_cluster.shape)
#all_data_to_cluster = all_data_to_cluster.iloc[0:100902, :]
#all_data_to_cluster = all_data_to_cluster.iloc[0:my_range, 1:all_data_to_cluster.shape[1]]
print(all_data_to_cluster.shape)

for number_of_pc in [21, 36, 48, 53, 62]:
    start_time = time.time()
    print(start_time  )
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold).fit(all_data_to_cluster.iloc[:, 0:number_of_pc])
    #print(clustering)
    AgglomerativeClustering()

    elapsed_time = time.time() - start_time
    print("time " + str(elapsed_time))
    print(clustering.labels_)

    #hs = homogeneity_score(identity.iloc[:, 1],clustering.labels_)
    #print(hs)

    file_object = open(
        'PCA_half_res/PCA_'+ str(number_of_pc) + '_AgglomerativeClustering' + '_' + str(distance_threshold) + '_' + file_name + '.csv', 'w')
    for b in range(len(clustering.labels_)):
        file_object.write(str(identity.iloc[b, 1]) + ',' + str(identity.iloc[b, 2]) + ',' + str(clustering.labels_[b]) + '\n')
    file_object.close()


    compareAB(identity.iloc[:, 1],clustering.labels_)

'''
print(clustering.labels_)
hs = homogeneity_score(all_data_to_cluster.iloc[:, 1], clustering.labels_)
print(hs)
cs = completeness_score(all_data_to_cluster.iloc[:, 0], all_data_to_cluster.iloc[:, 1])
print(cs)
vms = metrics.v_measure_score(all_data_to_cluster.iloc[:, 0], all_data_to_cluster.iloc[:, 1])
print(vms)
fowlkes_mallows_score = metrics.fowlkes_mallows_score(all_data_to_cluster.iloc[:, 1], clustering.labels_)
print(fowlkes_mallows_score)

'''


'''
NO PCA
adjusted_rand_score 0.4932543458730346
homogeneity_score 0.9914953739546918
completeness_score 0.8768722863161079
v_measure_score 0.9306678058273421
fowlkes_mallows_score 0.5632685872324349

PCA 21
adjusted_rand_score 0.6063885594427506
homogeneity_score 0.9382993042319508
completeness_score 0.8970654172336865
v_measure_score 0.917219173929373
fowlkes_mallows_score 0.6170450105440214

PCA 36
adjusted_rand_score 0.6264942306428517
homogeneity_score 0.9797239436362681
completeness_score 0.9021617237963031
v_measure_score 0.9393444640462751
fowlkes_mallows_score 0.6617990530276021


PCA 48
adjusted_rand_score 0.5525039310484637
homogeneity_score 0.9878742453052756
completeness_score 0.8886761262231859
v_measure_score 0.9356532825692183
fowlkes_mallows_score 0.6073244617204558

PCA 53
adjusted_rand_score 0.5233922891988586
homogeneity_score 0.9898449117834145
completeness_score 0.8830535184903942
v_measure_score 0.9334046288696721
fowlkes_mallows_score 0.5853879373764063

PCA 62
adjusted_rand_score 0.4954696247896687
homogeneity_score 0.9914601820532852
completeness_score 0.8774963928952572
v_measure_score 0.9310036895586901
fowlkes_mallows_score 0.5648822229773126
'''