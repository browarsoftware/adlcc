from keras.models import load_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import cv2

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# load the model
# path_to_img = 'd:\\Projects\\Python\\PycharmProjects\\celeba-dataset\\img_celeba\\'
#path_to_img = 'd:\\Projects\\Python\\PycharmProjects\\celeba-dataset\\same_twarze\\'
path_to_img = 'd:\\Projects\\Python\\PycharmProjects\\celeba-dataset\\img_celeba_network\\'

model = load_model('facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)

import mtcnn

from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
# demonstrate face detection on 5 Celebrity Faces Dataset
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


# print version
print(mtcnn.__version__)

import pandas as pd

data = pd.read_csv("d:\\Projects\\Python\\PycharmProjects\\celeba-dataset\\identity_CelebA.txt", sep=' ', header=None)

# pyplot.imshow(face)
# pyplot.show()
a = 0
# face = extract_face(detector, path_to_img + data[0][a])

import string

print(data.shape[0])
#for a in range(1):  # range(data.shape[0]):
for a in range(data.shape[0]):
    # aa = 197
    try:
        print(str(a) + " of " + str(data.shape[0]))
        # filename = 'd:\\Projects\\Python\\PycharmProjects\\celeba-dataset\\img_celeba_network\\' + data[0][a]

        #dd = data[0][a].replace(".jpg", ".png")
        #print(dd)
        #filename = path_to_img + dd
        filename = path_to_img + data[0][a]

        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        image = asarray(image)

        #print(filename)

        #image = cv2.imread(filename)
        #print(image.shape)
        image = cv2.resize(image, dsize=(160, 160))
        embedding = get_embedding(model, image)

        file_object = open('same_twarze_faces_data_network_OK.txt', 'a')
        #file_object.write(str(data[0][a]) + ',' + str(data[1][a]))
        file_object.write(str(data[0][a]) + ',' + str(data[1][a]))
        for b in range(embedding.shape[0]):
            file_object.write(',' + str(embedding[b]))
        file_object.write('\n')
        # Close the file
        file_object.close()

        #print(image.shape)
        '''
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        face = asarray(image)
        embedding = get_embedding(model, face)
        # print(embedding.shape[0])
        file_object = open('same_twarze_faces_data.txt', 'a')
        file_object.write(str(data[0][a]) + ',' + str(data[1][a]))
        for b in range(embedding.shape[0]):
            file_object.write(',' + str(embedding[b]))
        file_object.write('\n')
        # Close the file
        file_object.close()
        '''
    except FileNotFoundError:
        cc = 1

cv2.waitKey()
cv2.destroyAllWindows()