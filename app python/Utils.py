import os
import json
import imageio as iio
from threading import Thread
import numpy as np
import tensorflow as tf
import cv2


mainPath = "../hand_labels"
trainDataPath = mainPath + "/manual_train"
testDataPath = mainPath + "/manual_test"
baseShape = (1280, 720, 3)
newBaseShape = (1080, 1920, 3)
resized_shape = (224,224)

#What i nedd from the data AKA clasees
#1 - hand_box_center
# 2- is_left
# 3 - hand_pts

def __addData(startIndex, endIndex, labels, images, path):
    files = os.listdir(path)
    for i in range(startIndex, endIndex):
        fileName = files[i]
        if (fileName.rpartition(".")[2] == "jpg"):
            f = os.path.join(path, fileName)
            jsonPath = f.rsplit('.', 1)[0]  + "." + "json"

            img = np.array(iio.v3.imread(f))
            img_shape = img.shape 
            img_resized = tf.image.resize(img.copy(),resized_shape,method='nearest')
            
            json_file_open = open(jsonPath)
            json_file = json.load(json_file_open)["hand_pts"]
            json_file_open.close()
            
            for dim in json_file:
                newDim = lambda ind,x :(resized_shape[0]/img_shape[x]) * dim[ind]
                dim[0] = newDim(0,1)
                dim[1] = newDim(1,0)
                
            images.append(img_resized)
            labels.append(np.array(json_file))
            
        # if (fileName.rpartition(".")[2] == "json"):
        #     fOPen = open(f)
        #     data = json.load(fOPen)
        #     fOPen.close()

        #     # currentData = {}
        #     # currentData["hand_box_center"] = np.array(data["hand_box_center"])
        #     # currentData["is_left"] = data["is_left"]
        #     # currentData["hand_pts"] = np.array(data["hand_pts"])
            
        #     labels.append(np.array(data["hand_pts"]))
        # else:
        #     img = np.array(iio.v3.imread(f))

        #     img_after_padding = tf.image.resize(img, (244, 244), method='nearest') /255
        #     images.append(img_after_padding)
            # img = np.array(iio.v3.imread(f))
            # print(img.shape == newBaseShape)                 
            # if(img.shape == newBaseShape):
            #     i += 1
            # else:

            #     img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                
            #     img_shape = img.shape 
    
            #     get_padding_lenght = lambda current_index : (baseShape[current_index] - img_shape[current_index])
    
            #     img_after_padding = np.pad(img,pad_width=((0,get_padding_lenght(0)),
            #                                                (0,get_padding_lenght(1))
            #                                              ))
            #     img_after_blur = cv2.GaussianBlur(img_after_padding, (9,9), 500)
            #     images.append(img_after_blur) 

def __getBaseData(path, threadCount, countInEachThread,size):
    labels = []
    images = []
    threads = []

    for i in range(threadCount):
        if(size *2 <= i * countInEachThread or size <= len(labels) or size <= len(images)):
            break;
    
        th = Thread(name=f"data {i}", target=__addData, args=([i * countInEachThread, (i + 1) * countInEachThread,
                                                            labels, images, path]))
        th.start()
        threads.append(th)

    for thread in threads:
        thread.join()

    return np.array(labels)[:size,:],np.array(images)[:size,:]

def get_train_data(half_data=True,size=1912):
    # 3824 files on 16 thread with 239 each
    countInEachThread = 16
    threadCount = 239
    if(half_data) :
        countInEachThread = 478
        threadCount = 4

    labels, images = __getBaseData(trainDataPath, threadCount, countInEachThread,size)
    training_labels, training_images = labels, images
    return training_labels, training_images


def get_test_data(half_data=True,size=846):
    # 1692 files on 16 thread with 239 each
    countInEachThread = 12
    threadCount = 141
    if (half_data):
        countInEachThread = 94
        threadCount = 9


    labels, images = __getBaseData(testDataPath, threadCount, countInEachThread,size)
    test_labels, test_images = labels, images
    return test_labels, test_images