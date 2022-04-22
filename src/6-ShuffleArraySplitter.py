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
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data

    x_benign = open("/home/ahmed/Documents/Projects/IoT_Defence/DS/BenignDense.pkl","rb")
    x_gafgyt = open("/home/ahmed/Documents/Projects/IoT_Defence/DS/malware_graph_gafgyt_filteredDense.pkl","rb")
    x_mirai = open("/home/ahmed/Documents/Projects/IoT_Defence/DS/malware_graph_mirai_filteredDense.pkl","rb")
    x_tsunami = open("/home/ahmed/Documents/Projects/IoT_Defence/DS/malware_graph_tsunami_filteredDense.pkl","rb")
    # x_benign = open("/home/ahmed/Documents/Projects/IoT_Defence/DS/BenignLevels.pkl","rb")
    # x_gafgyt = open("/home/ahmed/Documents/Projects/IoT_Defence/DS/malware_graph_gafgyt_filteredLevels.pkl","rb")
    # x_mirai = open("/home/ahmed/Documents/Projects/IoT_Defence/DS/malware_graph_mirai_filteredLevels.pkl","rb")
    # x_tsunami = open("/home/ahmed/Documents/Projects/IoT_Defence/DS/malware_graph_tsunami_filteredLevels.pkl","rb")
    labels = [0,1,2,3]
    u = pickle._Unpickler(x_benign)
    u.encoding = 'latin1'
    x_benign = u.load()
    u = pickle._Unpickler(x_gafgyt)
    u.encoding = 'latin1'
    x_gafgyt = u.load()
    u = pickle._Unpickler(x_mirai)
    u.encoding = 'latin1'
    x_mirai = u.load()
    u = pickle._Unpickler(x_tsunami)
    u.encoding = 'latin1'
    x_tsunami = u.load()
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    print(len(x_benign))
    x_all = [x_benign,x_gafgyt,x_mirai,x_tsunami]
    sizes = [len(x_benign)/10,len(x_gafgyt)/10,len(x_mirai)/10,len(x_tsunami)/10]

    # ####### Generate Shuffle Array #######
    # ShuffleArray = []
    # for i in range(len(sizes)):
    #     listToShuffle = list(range(0, int(sizes[i])))
    #     random.shuffle(listToShuffle)
    #     ShuffleArray.append(listToShuffle)
    #
    # ShuffleToStore = open('/home/ahmed/Documents/Projects/IoT_Defence/ShuffleINIT/Shuffle.pkl', 'wb')
    # pickle.dump(ShuffleArray, ShuffleToStore)
    # exit()
    # ######################################

    ######## Restore Shuffle Array #######
    ShuffleArray = open("/home/ahmed/Documents/Projects/IoT_Defence/ShuffleINIT/Shuffle.pkl","rb")
    u = pickle._Unpickler(ShuffleArray)
    u.encoding = 'latin1'
    ShuffleArray = u.load()
    ######################################
    ShuffleTestPart = []
    for i in range(len(ShuffleArray)):
        ShuffleTestPart.append([])
        for j in range(len(ShuffleArray[i])):
            if j >= 0.8*sizes[i]:
                ShuffleTestPart[i].append(ShuffleArray[i][j])
        print(len(ShuffleTestPart[i]))

    x_trainFile = open('/home/ahmed/Documents/Projects/IoT_Defence/ShuffleINIT/BenignTestElementsShuffle.pkl', 'wb')
    pickle.dump(ShuffleTestPart[0], x_trainFile)
    x_trainFile = open('/home/ahmed/Documents/Projects/IoT_Defence/ShuffleINIT/GafgytTestElementsShuffle.pkl', 'wb')
    pickle.dump(ShuffleTestPart[1], x_trainFile)
    x_trainFile = open('/home/ahmed/Documents/Projects/IoT_Defence/ShuffleINIT/MiraiTestElementsShuffle.pkl', 'wb')
    pickle.dump(ShuffleTestPart[2], x_trainFile)
    x_trainFile = open('/home/ahmed/Documents/Projects/IoT_Defence/ShuffleINIT/TsunamiTestElementsShuffle.pkl', 'wb')
    pickle.dump(ShuffleTestPart[3], x_trainFile)
    print(len(ShuffleTestPart[0]))

    exit()


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
