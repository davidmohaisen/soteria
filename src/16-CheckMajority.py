"""
This tutorial shows how to generate adversarial examples
using C&W attack in white-box setting.
The original paper can be found at:
https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

import logging
import os
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import MadryEtAl
from cleverhans.loss import CrossEntropy
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import VirtualAdversarialMethod
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import train, model_eval, tf_model_load, model_argmax
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from six.moves import xrange
from sklearn.model_selection import train_test_split
import keras
from cleverhans.attacks import SPSA
from cleverhans.attacks import SaliencyMapMethod
import pickle
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ['CUDA_VISIBLE_DEVICES'] = ''


FLAGS = flags.FLAGS

LEARNING_RATE = .002
CW_LEARNING_RATE = .2
ATTACK_ITERATIONS = 1000


def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=True, nb_epochs=6,
                      batch_size=50, source_samples=10,
                      learning_rate=LEARNING_RATE,
                      attack_iterations=ATTACK_ITERATIONS,
                      model_path=os.path.join("models", "mnist"),
                      targeted=True):
    """
    MNIST tutorial for Carlini and Wagner's attack
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sessLevel = tf.Session()
    sessDense = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    x_test_Level = open("/home/ahmed/Documents/Projects/IoT_Defence/pkl/x_test_Levels.pkl","rb")
    y_test_Level = open("/home/ahmed/Documents/Projects/IoT_Defence/pkl/y_test_Levels.pkl","rb")
    x_test_Dense = open("/home/ahmed/Documents/Projects/IoT_Defence/pkl/x_test_Dense.pkl","rb")
    y_test_Dense = open("/home/ahmed/Documents/Projects/IoT_Defence/pkl/y_test_Dense.pkl","rb")
    labels = [0,1,2,3]
    u = pickle._Unpickler(x_test_Level)
    u.encoding = 'latin1'
    x_test_Level = u.load()
    u = pickle._Unpickler(y_test_Level)
    u.encoding = 'latin1'
    y_test_Level = u.load()
    u = pickle._Unpickler(x_test_Dense)
    u.encoding = 'latin1'
    x_test_Dense = u.load()
    u = pickle._Unpickler(y_test_Dense)
    u.encoding = 'latin1'
    y_test_Dense = u.load()

    x_test_Level = np.asarray(x_test_Level)
    y_test_Level = np.asarray(y_test_Level)
    x_test_Dense = np.asarray(x_test_Dense)
    y_test_Dense = np.asarray(y_test_Dense)

    x_test_Level = x_test_Level.reshape((len(x_test_Level),500,1))

    x_test_Dense = x_test_Dense.reshape((len(x_test_Dense),500,1))


    print(x_test_Level.shape)
    print(y_test_Level.shape)
    print(x_test_Dense.shape)
    print(y_test_Dense.shape)



    # Obtain Image Parameters
    img_rows, img_cols = x_test_Level.shape[1:3]
    nb_classes = 4

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    nb_filters = 64

    # Define TF model graph
    model = ModelBasicCNN(nb_classes, nb_filters)
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Load the model using TensorFlow
    ###########################################################################
    saver1 = tf.train.Saver()
    saver2 = tf.train.Saver()
    saver1.restore(sessLevel, "/home/ahmed/Documents/Projects/IoT_Defence/model/2-4Features500Levels.ckpt")
    saver2.restore(sessDense, "/home/ahmed/Documents/Projects/IoT_Defence/model/2-4Features500Dense.ckpt")
    print("Model restored.")

    predictsLevels = model_argmax(sessLevel,x,preds,x_test_Level)
    predictsDense = model_argmax(sessDense,x,preds,x_test_Dense)
    AccLevels = 0
    AccDense = 0
    TrueMajorityLabels = []
    PredictedMajorityLabels = []
    ToAdd = 0
    while ToAdd < len(y_test_Level) :
        TrueMajorityLabels.append(y_test_Level[ToAdd])
        ToAdd += 10
    print(len(TrueMajorityLabels))

    for i in range(len(TrueMajorityLabels)):
        PredictedMajorityLabels.append([0]*4)
        for j in range(10):
            PredictedMajorityLabels[i][predictsLevels[int((i*10)+j)]] += 1
            PredictedMajorityLabels[i][predictsDense[int((i*10)+j)]] += 1

    ErrorTotal = 0
    Predicts = []
    for i in range(len(TrueMajorityLabels)):
        PredClass = PredictedMajorityLabels[i].index(max(PredictedMajorityLabels[i]))
        Predicts.append(PredClass)
        if TrueMajorityLabels[i] != PredClass:
            ErrorTotal+=1
    print(1-(ErrorTotal/len(TrueMajorityLabels)))

    errorCount = [0]*4
    SizesAll = [0]*4
    BenignFalse = 0
    AllNoneBenign = 0
    labelsNames = ["Benign","Gafgyt","Mirai","Tsunami"]
    MiraiError = [0,0,0,0]
    for i in range(len(TrueMajorityLabels)):
        if TrueMajorityLabels[i] != Predicts[i]:
            errorCount[TrueMajorityLabels[i]]+=1
        SizesAll[TrueMajorityLabels[i]]+=1
        if Predicts[i] == 0 and TrueMajorityLabels[i] != Predicts[i] :
            BenignFalse += 1
        if TrueMajorityLabels[i] != 0 :
            AllNoneBenign += 1
        if Predicts[i] != 2 and TrueMajorityLabels[i] == 2 :
            MiraiError[Predicts[i]] += 1
    for i in range(len(errorCount)):
        print(labelsNames[i] + " Accuracy : " + str(1-float(errorCount[i]/SizesAll[i])))
    print("To Benign Error : "+str(float(BenignFalse/AllNoneBenign)))
    print("Mirai Class Error is : ",MiraiError)
def main(argv=None):
    mnist_tutorial_cw(viz_enabled=FLAGS.viz_enabled,
                      nb_epochs=FLAGS.nb_epochs,
                      batch_size=FLAGS.batch_size,
                      source_samples=FLAGS.source_samples,
                      learning_rate=FLAGS.learning_rate,
                      attack_iterations=FLAGS.attack_iterations,
                      model_path=FLAGS.model_path,
                      targeted=FLAGS.targeted)


if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', False, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
    flags.DEFINE_integer('source_samples', 1091, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                       'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_integer('attack_iterations', ATTACK_ITERATIONS,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', False,
                         'Run the tutorial in targeted mode?')

    tf.app.run()
