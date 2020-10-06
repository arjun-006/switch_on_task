import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Lambda, Input
from tensorflow.keras.models import Model
import numpy as np

# This python file is used for running inference

def visualize(img1,img2):
    """
    The thod is for visualising the difference between the two images.
    :param img1:
    :param img2:
    :return:
    """
    difference = cv2.subtract(img2, img1)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]
    img2[mask != 255] = [0, 0, 255]
    cv2.imshow("The difference",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./output.png",img2)

def get_siamese_net(input_shape):
    """
    Siamese network with Resnet50.
    Absolute difference of the feature vectors are calculated
    a classification is done with a Dense layer.
    :param input_shape:
    :return: siamese model
    """
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    model = ResNet50(weights='imagenet',include_top=False)
    print (model.summary())
    for layer in model.layers:
        layer.trainable = False
    features = model.get_layer("conv5_block3_out").output
    features = Flatten()(features)
    m = Model(inputs=model.input, outputs=features)
    features1 = m(left_input)
    features2 = m(right_input)
    L1_layer = Lambda(lambda tensors: abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([features1, features2])
    prediction = Dense(1,activation='sigmoid',name='visualized_layer')(L1_distance)
    siamese_net = Model(inputs=(left_input, right_input), outputs=prediction)
    return siamese_net

# Loading the images
img1 = cv2.imread("./good_image.png")
img2 = cv2.imread("./bad_image.png")

# Resizing the images so that it is compatible with Resnet50
img1 = cv2.resize(img1,(224,224),cv2.INTER_AREA)
img2 = cv2.resize(img2,(224,224),cv2.INTER_AREA)

# Preprocessing the images to make it compatible with Resnet50
left = tf.keras.applications.resnet50.preprocess_input(img1, data_format=None)
right = tf.keras.applications.resnet50.preprocess_input(img2, data_format=None)

# Getting the model
model = get_siamese_net((224,224,3))

# Loading the model weights
model.load_weights('./weights/weights.430.h5')

# Performing the prediction
res = model.predict([tf.expand_dims(left,axis=0),tf.expand_dims(right,axis=0)], steps=1, verbose=0)
print ("Prediction::",np.round(res)[0])
if np.round(res)[0] == 0.0:
    print ("The images are different")
    # img2 should be the image which has something missing
    # and img1 should be the expected image
    visualize(img1,img2)
else:
    print ("The images are similar")
