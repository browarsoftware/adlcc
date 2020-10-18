import pandas as pd
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn import metrics

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

    my_str = ars_str + "&" + hs_str + "&" + cs_str + "&" + vms_str + "&" + fms_str + "&" + sc_str
    return my_str




VVV =  pd.read_csv('PCA_half/Random_Second_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')

classes_count =len(pd.unique(VVV.iloc[:, 2]))
print('classes count = ' + str(classes_count))


strstr = 'NN_1024Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_.csv'
data = pd.read_csv('Supervised/' + strstr, header=None)
val_help = len(pd.unique(data.iloc[:, 2]))
CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:150].to_numpy())
print('SUPER NN=' + my_str + "&" + str(CNR))



VVV =  pd.read_csv('PCA_half/Random_Second_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')
strstr = 'SVM_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_.csv'
data = pd.read_csv('Supervised/' + strstr, header=None)
val_help = len(pd.unique(data.iloc[:, 2]))
CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:150].to_numpy())
print('SUPER SVM=' + my_str + "&" + str(CNR))




VVV =  pd.read_csv('PCA_half/Random_Second_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')
strstr = 'KNN_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_.csv'
data = pd.read_csv('Supervised/' + strstr, header=None)
val_help = len(pd.unique(data.iloc[:, 2]))



CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:150].to_numpy())
print('SUPER KNN=' + my_str + "&" + str(CNR))






number_of_pc = 62
VVV =  pd.read_csv('PCA_half/Random_Second_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')
strstr = 'NN_PCA_1024PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_.csv'
data = pd.read_csv('Supervised/' + strstr, header=None)
val_help = len(pd.unique(data.iloc[:, 2]))
CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:(3 + number_of_pc)].to_numpy())
print('SUPER NN PCA=' + my_str + "&" + str(CNR))


number_of_pc = 62
VVV =  pd.read_csv('PCA_half/Random_Second_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')
strstr = 'KNN_PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_.csv'
data = pd.read_csv('Supervised/' + strstr, header=None)
val_help = len(pd.unique(data.iloc[:, 2]))
CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:(3 + number_of_pc)].to_numpy())
print('SUPER KNN PCA=' + my_str + "&" + str(CNR))

number_of_pc = 62
VVV =  pd.read_csv('PCA_half/Random_Second_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')
strstr = 'SVM_PCAPCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_.csv'
data = pd.read_csv('Supervised/' + strstr, header=None)
val_help = len(pd.unique(data.iloc[:, 2]))
CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:(3 + number_of_pc)].to_numpy())
print('SUPER SVM PCA=' + my_str + "&" + str(CNR))





#TUTUTUTU

#for n_neighbors in [1,3,5]:
for n_neighbors in [1]:
    VVV =  pd.read_csv('PCA_half/Random_Second_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')

    strstr = 'knn=' + str(n_neighbors) + '_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_AgglomerativeClustering_34_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt.csv'
    data = pd.read_csv('PCA_half_class_res/' + strstr, header=None)

    val_help = len(pd.unique(data.iloc[:, 2]))
    CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
    my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:150].to_numpy())

    print('UNSPER KNN=' + str(n_neighbors) + my_str + "&" + str(CNR))



for n_neighbors in [1]:
    for number_of_pc in [36, 48, 53, 62]:
        VVV = pd.read_csv('PCA_half/PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')
        strstr = 'knn=' + str(n_neighbors) + '_PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_PCA_' + str(number_of_pc) + '_AgglomerativeClustering_34_PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt.csv'
        data = pd.read_csv('PCA_half_class_res/' + strstr, header=None)
        my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:(3 + number_of_pc)].to_numpy())
        val_help = len(pd.unique(data.iloc[:, 2]))
        CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
        print('UNSUPER KNN PCA=' + str(n_neighbors) + '_number_of_pc=' +str(number_of_pc) + my_str + "&" + str(CNR))


VVV =  pd.read_csv('PCA_half/PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')
strstr = 'LinearSVCRandom_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_AgglomerativeClustering_34_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt.csv.csv'
data = pd.read_csv('PCA_half_class_res/' + strstr, header=None)
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:150].to_numpy())
val_help = len(pd.unique(data.iloc[:, 2]))
CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
print('UNSUPER SVM' + my_str + "&" + str(CNR))

distance_threshold = 16
train_file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt"
valid_file_name = "PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK.txt"

for number_of_pc in [36, 48, 53, 62]:
    VVV = pd.read_csv('PCA_half/PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')

    strstr = 'LinearSVC_number_of_pc' + str(number_of_pc) + '.csv'
    data = pd.read_csv('PCA_half_class_res/' + strstr, header=None)
    my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:(3 + number_of_pc)].to_numpy())
    val_help = len(pd.unique(data.iloc[:, 2]))
    CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
    print('SVM_PCA' + str(number_of_pc) + my_str + "&" + str(CNR))

for neurons_count in [1024, 2048, 4096, 8192]:
    VVV = pd.read_csv('PCA_half/PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')

#    NN_1024Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_AgglomerativeClustering_34_Random_First_Half_same_twarze_faces_data_network_OK_CASIA - WebFace_align.txt.csv.csv
#    NN_1024Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_AgglomerativeClustering_34_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt.csv.csv

    cluster_file_name = 'NN_' + str(neurons_count) + 'Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt_AgglomerativeClustering_34_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt.csv.csv'
    data = pd.read_csv('PCA_half_class_res/' + cluster_file_name, header=None)
    my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:150].to_numpy())
    val_help = len(pd.unique(data.iloc[:, 2]))
    CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
    print('NN ' + str(neurons_count) + my_str + "&" + str(CNR))

train_file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"
valid_file_name = "PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"

for neurons_count in [1024, 2048, 4096, 8192]:
    for distance_threshold in [34]:
        for number_of_pc in [21, 36, 48, 53, 62]:
            VVV = pd.read_csv('PCA_half/PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt')

            cluster_file_name = 'PCA_' + str(number_of_pc) + '_AgglomerativeClustering' + '_' + str(
                distance_threshold) + '_' + train_file_name + '.csv'

            cluster_file_name = "NN_PCA" + str(neurons_count) + train_file_name + "_" + cluster_file_name + ".csv"
            data = pd.read_csv('PCA_half_class_res/' + cluster_file_name, header=None)
            my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2], VVV.iloc[:, 3:(3 + number_of_pc)].to_numpy())
            val_help = len(pd.unique(data.iloc[:, 2]))
            CNR = 1 - abs((classes_count - val_help) / (classes_count + val_help))
            print('NN ' + str(neurons_count) + " PCA" + str(number_of_pc) + my_str + "&" + str(CNR))

'''
cluster_file_name = 'LinearSVC_number_of_pc62PCA1-5.csv'
data = pd.read_csv('PCA_half_class_res/' + cluster_file_name, header=None)
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2])
print('SVM PCA 62 1-5 ' + str(neurons_count) + " PCA" + str(number_of_pc) + my_str)

cluster_file_name = 'LinearSVC_number_of_pc62PCA6-10.csv'
data = pd.read_csv('PCA_half_class_res/' + cluster_file_name, header=None)
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2])
print('SVM PCA 62 6-10 ' + str(neurons_count) + " PCA" + str(number_of_pc) + my_str)

cluster_file_name = 'LinearSVC_number_of_pc62PCA11-15.csv'
data = pd.read_csv('PCA_half_class_res/' + cluster_file_name, header=None)
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2])
print('SVM PCA 62 11-15 ' + str(neurons_count) + " PCA" + str(number_of_pc) + my_str)

cluster_file_name = 'LinearSVC_number_of_pc62PCA16-30.csv'
data = pd.read_csv('PCA_half_class_res/' + cluster_file_name, header=None)
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2])
print('SVM PCA 62 16-30 ' + str(neurons_count) + " PCA" + str(number_of_pc) + my_str)

cluster_file_name = 'NN_PCA2048PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt_PCA_21_AgglomerativeClustering_16_PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt.csv.csv'
data = pd.read_csv('PCA_half_class_res/' + cluster_file_name, header=None)
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2])
print('NN PCA 2048' + my_str)

strstr = 'knn=1_PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt_PCA_36_AgglomerativeClustering_16_PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt.csv'
data = pd.read_csv('PCA_half_class_res/' + strstr, header=None)
my_str = compareAB(data.iloc[:, 1], data.iloc[:, 2])
print(my_str)
'''