from tensorflow.keras.applications import ResNet50
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Lambda, Input, Conv2D
from tensorflow.keras.backend import abs
from tensorflow.keras.optimizers import Adam
import time
import random
import cv2
import os


# Model parameters
train_data1 = './dataset/train_img1/'
train_data2 = './dataset/train_img2/'
batch_size = 5
n_iter = 500
evaluate_every = 10
weights_path = "./weights/saved_weights/"



def train_feeder():
    """
    Custom input pipeline for the simaese network
    :return:
    """
    num1 = random.randint(0, 5000)
    while not os.path.isfile(os.path.join(train_data1,str(num1) + ".png")):
        num1 = random.randint(0, 5000)

    if num1 > 2000:
        prediction = 0.0
    else:
        prediction = 1.0
    # print ("input filename::", str(num1) + ".png")
    # print("Target::", str(prediction))
    left = cv2.imread(os.path.join(train_data1 + str(num1) + ".png"))
    right = cv2.imread(os.path.join(train_data2 + str(num1) + ".png"))
    left = tf.keras.applications.resnet50.preprocess_input(left, data_format=None)
    right = tf.keras.applications.resnet50.preprocess_input(right, data_format=None)
    return left,right,prediction



def get_batch(batch_size):
    """
    Custom minibatch generator in the input pipeline
    :param batch_size:
    :return:
    """
    left = []
    right = []
    target = []
    for i in range(0,batch_size):
         l,r,t = train_feeder()
         left.append(l)
         right.append(r)
         target.append(t)
    return left,right,target


def generate(batch_size, s="train"):
    """a generator for batches, so model.fit_generator can be used. """
    while True:
        pairs,target = get_batch(batch_size)
        yield (pairs,target)


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


model_siamese = get_siamese_net((224, 224, 3))
optimizer = Adam(lr = 0.0001)
model_siamese.compile(loss="binary_crossentropy",optimizer=optimizer)
print (model_siamese.summary())



print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
    print ("Iteration::",i)
    left, right, target = get_batch(batch_size)


    loss = model_siamese.train_on_batch([left,right], target)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time() - t_start) / 60.0))
        print("Train Loss: {0}".format(loss))
        model_siamese.save_weights(os.path.join(weights_path, 'weights.{}.h5'.format(i)))

# This code was used to compare the performance of other networks for feature extraction

# cosine_similarity = tf.keras.losses.CosineSimilarity(axis =1)
# sess = tf.Session()
# sim = cosine_similarity(features2,features1).eval(session = sess)
# print (sim)







