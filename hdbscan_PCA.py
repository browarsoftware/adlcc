import hdbscan
import pandas as pd
from sklearn.metrics.cluster import homogeneity_score
import time
import datetime

#file_name = "30-16faces.data.attr.csv"
file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt"
min_cluster_size = 2


all_data_to_cluster = pd.read_csv("PCA_half/" + file_name).iloc[:, 3:150]
#print(all_data_to_cluster.head())

identity = pd.read_csv("PCA_half/PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt")
#print(all_data_to_cluster.head)
#print(all_data_to_cluster.shape)
#print(identity)

#12 - 0.32
#21 - 0.51
#36 - 0.76
#48 - 0.91
#53 - 0.95
#62 - 0.999


#min_cluster_size = 2
#number_of_pc = 12
#2020-06-20 23:58:34.769655
#time 722.4805014133453
#0.3371806220763438

#number_of_pc = 128
#2020-06-21 09:59:30.374274
#time 6916.851599931717
#0.783042336247379
for min_cluster_size in [2,3,4,5]:
    for number_of_pc in [21, 36, 48, 53, 62]:
        print("number_of_pc = " + str(number_of_pc))
        print(datetime.datetime.now())
        clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size)
        start_time = time.time()

        clusterin_labels = clusterer.fit_predict(all_data_to_cluster.iloc[:,0:number_of_pc])
        elapsed_time = time.time() - start_time
        print("time " + str(elapsed_time))

        hs = homogeneity_score(identity.iloc[:, 1],clusterin_labels)

        print(hs)

        file_object = open('PCA_half_res/PCA_' + str(number_of_pc) + '_hdbscan' + '_' + str(min_cluster_size) + '_' + file_name + '.csv', 'w')
        for b in range(len(clusterin_labels)):
            file_object.write(str(identity.iloc[b, 1]) + ',' + str(identity.iloc[b, 2]) + ',' + str(clusterin_labels[b]) + '\n')
        file_object.close()




    '''
    cluster_size = 2
    
    number_of_pc = 21
    2020-06-21 13:58:09.953910
    time 1335.7572674751282
    0.5578801752997145
    number_of_pc = 36
    2020-06-21 14:20:28.971188
    time 3130.768526315689
    0.7334417113011455
    number_of_pc = 48
    2020-06-21 15:12:50.029590
    time 4829.7171022892
    0.7714590951839401
    number_of_pc = 53
    2020-06-21 16:33:22.986362
    time 4952.57829284668
    0.7755518620158005
    number_of_pc = 62
    2020-06-21 17:55:58.740617
    time 3238.025433778763
    0.7660307431668031
    number_of_pc = 128
    2020-06-21 18:50:07.115690
    time 6940.758038282394
    0.6727701408783695
    '''


    '''
    #for min_cluster_size in [10]:#range(10, 10):
    for min_cluster_size in [10]:#range(10, 10):
        clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size)
        start_time = time.time()
        print(min_cluster_size)
        print(start_time)
        clusterin_labels = clusterer.fit_predict(all_data_to_cluster.iloc[:, 2:130])
        elapsed_time = time.time() - start_time
        print("time " + str(elapsed_time))
        #print(clusterin_labels)
    
    
        hs = homogeneity_score(all_data_to_cluster.iloc[:, 1],clusterin_labels)
    
        print(hs)
    
        file_object = open('hdbscan' + '_' + str(min_cluster_size) + '_' + file_name + '.csv', 'w')
        for b in range(len(clusterin_labels)):
            file_object.write(str(all_data_to_cluster.iloc[b, 0]) + ',' + str(all_data_to_cluster.iloc[b, 1]) + ',' + str(clusterin_labels[b]) + '\n')
        file_object.close()
    
    '''

'''
3 and 4


D:\Projects\Python\PycharmProjects\DLIB_Pytorch\venv\Scripts\python.exe D:/Projects/Python/PycharmProjects/DLIB_Pytorch_face_recognition/Clustering/hdbscan_PCA.py
number_of_pc = 21
2020-06-22 01:15:09.354611
time 1234.5099122524261
0.49786806630963076
number_of_pc = 36
2020-06-22 01:35:46.954188
time 2961.597568511963
0.6994409922611685
number_of_pc = 48
2020-06-22 02:25:18.701426
time 4511.991073846817
0.7482601650138988
number_of_pc = 53
2020-06-22 03:40:33.812254
time 4424.574028253555
0.7557060302508163
number_of_pc = 62
2020-06-22 04:54:28.598815
time 3248.8782851696014
0.7917132723483848
number_of_pc = 128
2020-06-22 05:48:47.659936
time 6928.315562725067
0.7918281014406385
number_of_pc = 21
2020-06-22 07:44:18.991943
time 1067.2419233322144
0.4492466753376207
number_of_pc = 36
2020-06-22 08:02:09.225560
time 2406.7553918361664
0.6600142437618362
number_of_pc = 48
2020-06-22 08:42:18.946278
time 3901.40434718132
0.7145750124475468
number_of_pc = 53
2020-06-22 09:47:23.419320
time 4526.0901601314545
0.7231004295772196
number_of_pc = 62
2020-06-22 11:02:52.634519
time 3231.8871161937714
0.7592089316879265
number_of_pc = 128
2020-06-22 11:56:54.791664
time 7118.346848964691
0.7591621905018694

'''
