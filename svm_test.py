import pandas as pd

def append_to_file(file_path, string_to_save):
    file_object = open(file_path, 'a')
    file_object.write(string_to_save + '\n')
    file_object.close()


n_neighbors = 1
distance_threshold = 16
train_file_name = "Random_First_Half_same_twarze_faces_data_network_OK.txt"
valid_file_name = "Random_Second_Half_same_twarze_faces_data_network_OK.txt"
cluster_file_name = "AgglomerativeClustering_16_Random_First_Half_same_twarze_faces_data_network_OK.txt.csv"



train_data = pd.read_csv('PCA_half/' + train_file_name)
valid_data = pd.read_csv('PCA_half/' + valid_file_name)
clustering_data = pd.read_csv('PCA_half_res/' + cluster_file_name, header=None)

class_id = clustering_data.iloc[:, 2].to_numpy()
X = train_data.iloc[:, 3:150].to_numpy()
X_valid = valid_data.iloc[:, 3:150].to_numpy()
id_true_valid = valid_data.iloc[:, 2].to_numpy()



from sklearn.svm import LinearSVC
import pickle
#model = SVC(kernel='linear', probability=True, verbose=True)
model = LinearSVC(verbose=True)


print('Fitting model')
model.fit(X, class_id)
filename = 'models_res/LinearSVC' + train_file_name + '_' + cluster_file_name + '.sav'
pickle.dump(model, open(filename, 'wb'))

file_name_res = 'LinearSVC' + train_file_name + '_' + cluster_file_name + '.csv'

#loaded_model = pickle.load(open(filename, 'rb'))
# test model on a random example from the test dataset

for index_help in range(id_true_valid.shape[0]):
    xxx = X_valid[index_help:(index_help + 1), ]
    pred_id = model.predict(xxx)[0]
    #pred_id2 = loaded_model.predict(xxx)[0]

    #print(str(pred_id) + "=" + str(pred_id2))

    strstr = str(valid_data.iloc[index_help, 0]) + "," + str(id_true_valid[index_help]) + "," + str(pred_id)
    # print(strstr)
    if (index_help % 1000 == 0):
        print(index_help)
    append_to_file('PCA_half_class_res/' + file_name_res, strstr)


#####################################################################################

'''
from sklearn.svm import LinearSVC
import pickle

train_file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt"
valid_file_name = "PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK.txt"

distance_threshold = 16
train_data = pd.read_csv('PCA_half/' + train_file_name)
valid_data = pd.read_csv('PCA_half/' + valid_file_name)

id_true_valid = valid_data.iloc[:, 2].to_numpy()

for number_of_pc in [36, 48, 53, 62]:
    cluster_file_name = 'PCA_' + str(number_of_pc) + '_AgglomerativeClustering' + '_' + str(
        distance_threshold) + '_' + train_file_name + '.csv'
    clustering_data = pd.read_csv('PCA_half_res/' + cluster_file_name, header=None)
    class_id = clustering_data.iloc[:, 2].to_numpy()

    X = train_data.iloc[:, 3:150]
    X_valid = valid_data.iloc[:, 3:150]

    X = X.iloc[:, 0:number_of_pc].to_numpy()
    X_valid = X_valid.iloc[:, 0:number_of_pc].to_numpy()

    #print(X.shape)
    #print(X_valid.shape)

    #neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    #neigh.fit(X, class_id)
    #file_name = 'knn=' + str(n_neighbors) + "_" + train_file_name + "_" + cluster_file_name

    print('Fitting model')
    model = LinearSVC(verbose=True)
    model.fit(X, class_id)
    filename = 'models_res/LinearSVC_number_of_pc' + str(number_of_pc) + train_file_name + '_' + cluster_file_name + '.sav'
    pickle.dump(model, open(filename, 'wb'))

    file_name_res = 'LinearSVC_number_of_pc' + str(number_of_pc) + train_file_name + '_' + cluster_file_name + '.csv'

    for index_help in range(id_true_valid.shape[0]):
        xxx = X_valid[index_help:(index_help + 1), ]
        pred_id = model.predict(xxx)[0]
        # pred_id2 = loaded_model.predict(xxx)[0]

        # print(str(pred_id) + "=" + str(pred_id2))

        strstr = str(valid_data.iloc[index_help, 0]) + "," + str(id_true_valid[index_help]) + "," + str(pred_id)
        # print(strstr)
        if (index_help % 1000 == 0):
            print(index_help)
        append_to_file('PCA_half_class_res/' + file_name_res, strstr)
'''

from sklearn.svm import SVC

import pickle

train_file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK.txt"
valid_file_name = "PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK.txt"

distance_threshold = 16
train_data = pd.read_csv('PCA_half/' + train_file_name)
valid_data = pd.read_csv('PCA_half/' + valid_file_name)

id_true_valid = valid_data.iloc[:, 2].to_numpy()

for number_of_pc in [36, 48, 53, 62]:
    cluster_file_name = 'PCA_' + str(number_of_pc) + '_AgglomerativeClustering' + '_' + str(
        distance_threshold) + '_' + train_file_name + '.csv'
    clustering_data = pd.read_csv('PCA_half_res/' + cluster_file_name, header=None)
    class_id = clustering_data.iloc[:, 2].to_numpy()

    X = train_data.iloc[:, 3:150]
    X_valid = valid_data.iloc[:, 3:150]

    X = X.iloc[:, 0:number_of_pc].to_numpy()
    X_valid = X_valid.iloc[:, 0:number_of_pc].to_numpy()

    #print(X.shape)
    #print(X_valid.shape)

    #neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    #neigh.fit(X, class_id)
    #file_name = 'knn=' + str(n_neighbors) + "_" + train_file_name + "_" + cluster_file_name

    print('Fitting model')
    #model = LinearSVC(verbose=True)
    model = SVC(kernel='rbf', random_state=0, verbose=True)
    model.fit(X, class_id)
    filename = 'models_res/RBFSVC_number_of_pc' + str(number_of_pc) + train_file_name + '_' + cluster_file_name + '.sav'
    try:
        pickle.dump(model, open(filename, 'wb'), protocol=4)
    except:
        print(':-/')

    file_name_res = 'RBFSVC_number_of_pc' + str(number_of_pc) + train_file_name + '_' + cluster_file_name + '.csv'

    for index_help in range(id_true_valid.shape[0]):
        xxx = X_valid[index_help:(index_help + 1), ]
        pred_id = model.predict(xxx)[0]
        # pred_id2 = loaded_model.predict(xxx)[0]

        # print(str(pred_id) + "=" + str(pred_id2))

        strstr = str(valid_data.iloc[index_help, 0]) + "," + str(id_true_valid[index_help]) + "," + str(pred_id)
        # print(strstr)
        if (index_help % 1000 == 0):
            print(index_help)
        append_to_file('PCA_half_class_res/' + file_name_res, strstr)