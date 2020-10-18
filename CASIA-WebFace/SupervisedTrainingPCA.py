def append_to_file(file_path, string_to_save):
    file_object = open(file_path, 'a')
    file_object.write(string_to_save + '\n')
    file_object.close()

# multi-class classification with Keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.pipeline import Pipeline


# load dataset
distance_threshold = 16
#train_file_name = "Random_First_Half_same_twarze_faces_data_network_OK.txt"
train_file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"
#valid_file_name = "Random_Second_Half_same_twarze_faces_data_network_OK.txt"
valid_file_name = 'PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt'
#cluster_file_name = "AgglomerativeClustering_16_Random_First_Half_same_twarze_faces_data_network_OK.txt.csv"

train_data = pd.read_csv('PCA_half/' + train_file_name)
valid_data = pd.read_csv('PCA_half/' + valid_file_name)
#clustering_data = pd.read_csv('PCA_half_res/' + cluster_file_name, header=None)

#dataframe = pandas.read_csv("iris.data", header=None)
#dataset = dataframe.values

#class_id = clustering_data.iloc[:, 2].to_numpy()
class_id = train_data.iloc[:, 2].to_numpy()
#X = train_data.iloc[:, 3:150].to_numpy()
#X_valid = valid_data.iloc[:, 3:150].to_numpy()

number_of_pc = 62

X = train_data.iloc[:, 3:(3 + number_of_pc)].to_numpy()
X_valid = valid_data.iloc[:, 3:(3 + number_of_pc)].to_numpy()


id_true_valid = valid_data.iloc[:, 2].to_numpy()

#X = dataset[:, 0:4].astype(float)
#Y = dataset[:, 4]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(class_id)
encoded_Y = encoder.transform(class_id)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
dummy_len = len(np.unique(class_id))
print(dummy_y.shape)
#neurons_count = 1024#2048#4096#8192
neurons_count = 2048
print(X.shape)
print(class_id.shape)
# define baseline model
def baseline_model(xx):
    # create model
    model = Sequential()
    model.add(Dense(neurons_count, input_dim=number_of_pc, activation='relu'))
    model.add(Dense(xx, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


neurons_count = 1024



model = baseline_model(dummy_len)
model.fit(X, dummy_y, epochs=10, batch_size=64)
model.save("Supervised/NN_PCA" + str(neurons_count) + "_" + train_file_name + ".mod")



file_name_res = "NN_PCA_" + str(neurons_count) + train_file_name + "_.csv"




from keras import backend as K
for index_help in range(id_true_valid.shape[0]):
    #xxx = X_valid[index_help:(index_help + 1), ]
    #pred_id = model.predict(xxx)[0]

    #pred_id2 = loaded_model.predict(xxx)[0]

    #print(str(pred_id) + "=" + str(pred_id2))

    X_valid_1 = X_valid[index_help,]
    X_valid_1 = K.expand_dims(X_valid_1, 0)
    predictions = model.predict_classes(X_valid_1)



    strstr = str(valid_data.iloc[index_help, 0]) + "," + str(id_true_valid[index_help]) + "," + str(encoder.inverse_transform(predictions)[0])
    # print(strstr)
    if (index_help % 1000 == 0):
        print(index_help)
    append_to_file('Supervised/' + file_name_res, strstr)




file_name_res = "SVM_PCA" + train_file_name + "_.csv"
filename = 'Supervised/LinearSVC_PCA_HALF_DATA_TEST.sav'
from sklearn.svm import LinearSVC
import pickle

print('Fitting model')
#model = LinearSVC(verbose=True)
model = LinearSVC(verbose=True)
print(X.shape)
print(dummy_y.shape)
model.fit(X = X, y = encoded_Y)
try:
    pickle.dump(model, open(filename, 'wb'), protocol=4)
except:
    print(':-/')

model = pickle.load(open(filename, 'rb'))

#kneighbors_res = neigh.kneighbors(xxx)
#id_help = kneighbors_res[1][0]

for index_help in range(id_true_valid.shape[0]):
    xxx = X_valid[index_help:(index_help + 1), ]
    pred_id = model.predict(xxx)[0]
    #kneighbors_res = neigh.kneighbors(xxx)
    #id_help = kneighbors_res[1][0]
    strstr = str(valid_data.iloc[index_help, 0]) + "," + str(id_true_valid[index_help]) + "," + str(
        encoder.inverse_transform(np.asarray([pred_id]))[0])
    # print(strstr)
    if (index_help % 1000 == 0):
        print(index_help)
    append_to_file('Supervised/' + file_name_res, strstr)




file_name_res = "KNN_" + train_file_name + "_.csv"
from sklearn.svm import LinearSVC
import pickle

print('Fitting model')
#model = LinearSVC(verbose=True)
from sklearn.neighbors import KNeighborsClassifier
n_neighbors = 1
neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
neigh.fit(X, encoded_Y)

#kneighbors_res = neigh.kneighbors(xxx)
#id_help = kneighbors_res[1][0]

for index_help in range(id_true_valid.shape[0]):
    xxx = X_valid[index_help:(index_help + 1), ]
    pred_id = neigh.predict(xxx)[0]
    #kneighbors_res = neigh.kneighbors(xxx)
    #id_help = kneighbors_res[1][0]
    strstr = str(valid_data.iloc[index_help, 0]) + "," + str(id_true_valid[index_help]) + "," + str(
        encoder.inverse_transform(np.asarray([pred_id]))[0])
    # print(strstr)
    if (index_help % 1000 == 0):
        print(index_help)
    append_to_file('Supervised/' + file_name_res, strstr)