import hdbscan
import pandas as pd
from sklearn.metrics.cluster import homogeneity_score
import time

#file_name = "30-16faces.data.attr.csv"
#file_name = "faces.data.2020-06-18.csv"
file_name = "Random_First_Half_same_twarze_faces_data_network_OK.txt"

min_cluster_size = 3
#plot_clusters(data, hdbscan.HDBSCAN, (), {'min_cluster_size':15})


#all_data_to_cluster = pd.read_csv("30-16faces.data.attr.csv")
#all_data_to_cluster = all_data.iloc[1:10000, 2:130]

#all_data_to_cluster = pd.read_csv("29-2128faces.data.attr.csv")
#all_data_to_cluster = pd.read_csv("faces.data.2020-06-18.csv")

#10 0.7022974822866088
#5 0.7976066597038978
#2 0.7738660831646755, time: 719.9608981609344


#faces.data.2020-06-18.csv
#min_cluster_size = 2
#time 7112.385579824448
#0.783042336247379


#faces.data.2020-06-18.csv
#min_cluster_size = 2
#time 7147.074876070023
#0.7918281014406385

all_data_to_cluster = pd.read_csv('PCA_half/' + file_name)
for min_cluster_size in [2,3,4,5]:#range(10, 10):
    clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size)
    start_time = time.time()
    print(min_cluster_size)
    print(start_time)
    clusterin_labels = clusterer.fit_predict(all_data_to_cluster.iloc[:, 3:150])
    elapsed_time = time.time() - start_time
    print("time " + str(elapsed_time))
    #print(clusterin_labels)


    hs = homogeneity_score(all_data_to_cluster.iloc[:, 1],clusterin_labels)

    print(hs)

    file_object = open('PCA_half_res/hdbscan' + '_' + str(min_cluster_size) + '_' + file_name + '.csv', 'w')
    for b in range(len(clusterin_labels)):
        file_object.write(str(all_data_to_cluster.iloc[b, 0]) + ',' + str(all_data_to_cluster.iloc[b, 1]) + ',' + str(clusterin_labels[b]) + '\n')
    file_object.close()

'''

import hdbscan
from sklearn.datasets import make_blobs

data, _ = make_blobs(1000)

clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(data)
'''



'''
4
1592615437.87077
time 7126.012009859085
0.7591621905018694
5
1592622574.0548687
time 7155.487308740616
0.7265110529914304
6
1592629739.670439
time 7016.731696128845
0.6947361090107139
7
1592636759.3874319
time 6914.252791643143
0.6643526761903668
8
1592643676.8254576
time 7123.69651889801
0.6340565831116687
9
1592650803.450594
time 7216.432926654816
0.6044220716800135

Process finished with exit code 0



'''