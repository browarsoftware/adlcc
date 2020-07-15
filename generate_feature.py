#https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

# example of loading the keras facenet model
from keras.models import load_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import cv2

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# load the model
path_to_img = 'd:\\Projects\\Python\\PycharmProjects\\celeba-dataset\\img_celeba\\'
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

detector = MTCNN()



def extract_face(detector, filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    #detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

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



#pyplot.imshow(face)
#pyplot.show()
a = 0
#face = extract_face(detector, path_to_img + data[0][a])


print(data.shape[0])
for a in range(data.shape[0]):
    #aa = 197
    print(str(a) + " of " + str(data.shape[0]))
    #print('file:' + data[a][0])
    #print(path_to_img + data[0][a])
    try:
        face = extract_face(detector, path_to_img + data[0][a])
        cv2.imwrite('d:\\Projects\\Python\\PycharmProjects\\celeba-dataset\\img_celeba_network\\' + data[0][a], cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        #cv2.imshow('vv', face)




        '''
        embedding = get_embedding(model, face)
        print(embedding.shape[0])
        # Append 'hello' at the end of file
        file_object = open('sample.txt', 'a')
        file_object.write(str(data[0][a]) + ',' + str(data[1][a]))
        for b in range(embedding.shape[0]):
            file_object.write(',' + str(embedding[b]))
        file_object.write('\n')
        # Close the file
        file_object.close()
    '''
    except IndexError:
        cc = 1

    '''
    embedding = get_embedding(model, face)
    print(embedding.shape[0])
    # Append 'hello' at the end of file
    file_object = open('sample.txt', 'a')
    file_object.write(str(data[0][a]) + ',' + str(data[1][a]))
    for b in range(embedding.shape[0]):
        file_object.write(',' + str(embedding[b]))
    file_object.write('\n')
    # Close the file
    file_object.close()
    '''


