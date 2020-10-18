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
train_file_name = "Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"
valid_file_name = "Random_Second_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"
cluster_file_name = "AgglomerativeClustering_34_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt.csv"

train_data = pd.read_csv('PCA_half/' + train_file_name)
valid_data = pd.read_csv('PCA_half/' + valid_file_name)
clustering_data = pd.read_csv('PCA_half_res/' + cluster_file_name, header=None)

#dataframe = pandas.read_csv("iris.data", header=None)
#dataset = dataframe.values

class_id = clustering_data.iloc[:, 2].to_numpy()
X = train_data.iloc[:, 3:150].to_numpy()
X_valid = valid_data.iloc[:, 3:150].to_numpy()
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

#neurons_count = 1024#2048#4096#8192
neurons_count = 2048
print(X.shape)
print(class_id.shape)
# define baseline model
def baseline_model(xx):
    # create model
    model = Sequential()
    model.add(Dense(neurons_count, input_dim=128, activation='relu'))
    model.add(Dense(xx, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

for neurons_count in [1024, 2048, 4096, 8192]:
    model = baseline_model(dummy_len)
    model.fit(X, dummy_y, epochs=10, batch_size=64)
    model.save("models_res/NN_" + str(neurons_count) + "_" + train_file_name + "_" + cluster_file_name + ".mod")
    file_name_res = "NN_" + str(neurons_count) + train_file_name + "_" + cluster_file_name + ".csv"

    # make class predictions with the model
    predictions = model.predict_classes(X_valid)
    #res = model.predict(X_valid)

    for index_help in range(id_true_valid.shape[0]):
        #xxx = X_valid[index_help:(index_help + 1), ]
        #pred_id = model.predict(xxx)[0]

        #pred_id2 = loaded_model.predict(xxx)[0]

        #print(str(pred_id) + "=" + str(pred_id2))

        strstr = str(valid_data.iloc[index_help, 0]) + "," + str(id_true_valid[index_help]) + "," + str(predictions[index_help])
        # print(strstr)
        if (index_help % 1000 == 0):
            print(index_help)
        append_to_file('PCA_half_class_res/' + file_name_res, strstr)




'''
# load dataset
distance_threshold = 16
train_file_name = "Random_First_Half_same_twarze_faces_data_network_OK.txt"
valid_file_name = "Random_Second_Half_same_twarze_faces_data_network_OK.txt"
cluster_file_name = "AgglomerativeClustering_16_Random_First_Half_same_twarze_faces_data_network_OK.txt.csv"
train_data = pd.read_csv('PCA_half/' + train_file_name)
valid_data = pd.read_csv('PCA_half/' + valid_file_name)
clustering_data = pd.read_csv('PCA_half_res/' + cluster_file_name, header=None)
#dataframe = pandas.read_csv("iris.data", header=None)
#dataset = dataframe.values
class_id = clustering_data.iloc[:, 2].to_numpy()
X = train_data.iloc[:, 3:150].to_numpy()
X_valid = valid_data.iloc[:, 3:150].to_numpy()
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
'''

distance_threshold = 16
number_of_pc = 21

#neurons_count = 1024#2048#4096#8192
neurons_count = 2048

train_file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"
valid_file_name = "PCA_Transformed_Half_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"

train_data = pd.read_csv('PCA_half/' + train_file_name)
valid_data = pd.read_csv('PCA_half/' + valid_file_name)



file_name = "PCA_principalComponents_Random_First_Half_same_twarze_faces_data_network_OK_CASIA-WebFace_align.txt"
for neurons_count in [1024, 2048, 4096, 8192]:
    for distance_threshold in [34]:
        for number_of_pc in [21, 36, 48, 53, 62]:

            cluster_file_name = 'PCA_' + str(number_of_pc) + '_AgglomerativeClustering' + '_' + str(
                distance_threshold) + '_' + train_file_name + '.csv'
            clustering_data = pd.read_csv('PCA_half_res/' + cluster_file_name, header=None)

            # dataframe = pandas.read_csv("iris.data", header=None)
            # dataset = dataframe.values

            class_id = clustering_data.iloc[:, 2].to_numpy()
            X = train_data.iloc[:, 3:150]
            X_valid = valid_data.iloc[:, 3:150]
            id_true_valid = valid_data.iloc[:, 2].to_numpy()

            X = X.iloc[:, 0:number_of_pc].to_numpy()
            X_valid = X_valid.iloc[:, 0:number_of_pc].to_numpy()

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


            # encode class values as integers
            encoder = LabelEncoder()
            encoder.fit(class_id)
            encoded_Y = encoder.transform(class_id)
            # convert integers to dummy variables (i.e. one hot encoded)
            dummy_y = np_utils.to_categorical(encoded_Y)
            dummy_len = len(np.unique(class_id))


            model = baseline_model(dummy_len)
            model.fit(X, dummy_y, epochs=10, batch_size=64)
            model.save("models_res/NN_PCA" + str(neurons_count) + "_" + train_file_name + "_" + cluster_file_name + ".mod")
            file_name_res = "NN_PCA" + str(neurons_count) + train_file_name + "_" + cluster_file_name + ".csv"

            predictions = model.predict_classes(X_valid)

            for index_help in range(id_true_valid.shape[0]):
                strstr = str(valid_data.iloc[index_help, 0]) + "," + str(id_true_valid[index_help]) + "," + str(predictions[index_help])
                # print(strstr)
                if (index_help % 1000 == 0):
                    print(index_help)
                append_to_file('PCA_half_class_res/' + file_name_res, strstr)