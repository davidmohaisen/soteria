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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.platform import flags
import seaborn as sns
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
FLAGS = flags.FLAGS

LEARNING_RATE = .001
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
    x_test = np.loadtxt('/home/ahmed/Documents/sdn/dataVectors/test8VectorUpdated.txt')
    y_test = np.loadtxt('/home/ahmed/Documents/sdn/dataVectors/test8LabelsUpdated.txt')
    y_testBefore = y_test
    x_test = x_test.reshape((34717,68,1))
    y_test = keras.utils.to_categorical(y_test, 8)

    ###########################################################################################
    # Load the model using TensorFlow
    ###########################################################################################
    # Obtain Image Parameters
    img_rows, img_cols = x_test.shape[1:3]
    nb_classes = y_test.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    nb_filters = 68

    # Define TF model graph
    model = ModelBasicCNN(nb_classes, nb_filters)
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    print("Defined TensorFlow model graph.")

    saver = tf.train.Saver()
    saver.restore(sess, "/home/ahmed/Documents/sdn/model/UpdatedModel8.ckpt")
    print("Model restored.")
    ###########################################################################################


    toAccumulate = [0,1,4,10,12,14,16,32,33,36,42,44,48,50,51,54,60] # Else, averaging
    tcpRelated = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    udpRelated = [32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]
    icmpRelated = [50,51,52,53,54,55,56,57,58,59,60,61]
    outgoing = [1,63,7,9,21,23,25,27,29,31,33,65,39,41,51,67,57,59] # Else, incoming
    fractions = [62,63,64,65,66,67]
    TCPFractions = [62,63]
    UDPFractions = [64,65]
    ICMPFractions = [66,67]

    tcpIn = 0
    tcpOut = 1
    udpIn = 32
    udpOut = 33
    icmpIn = 50
    icmpOut = 51
    for itarget in range(8):
        print("Tergeted Class = " + str(itarget))
        confusion = []
        for i in range(8):
            confusion.append([])
            for j in range(8):
                confusion[i].append(0)
        for iStart in range(8):
            ###########################################################################################
            predicts = model_argmax(sess,x,preds,x_test)
            total = 0
            correct = 0
            toManipulate = []
            toMask = []
            maxBenignTCP = 0
            maxBenignUDP = 0
            maxBenignICMP = 0
            TCPBIndex = -1
            UDPBIndex = -1
            ICMPBIndex = -1
            maxMalTCP = 0
            maxMalUDP = 0
            maxMalICMP = 0
            TCPMIndex = -1
            UDPMIndex = -1
            ICMPMIndex = -1
            BenignTCP = []
            BenignUDP = []
            BenignICMP = []
            MalTCP = []
            MalUDP = []
            MalICMP = []
            ####################################
            TargettedClass = itarget
            FromClass = iStart
            ####################################
            for i in range(len(y_testBefore)):
                if y_testBefore[i] == TargettedClass :
                    total+=1
                    if predicts[i]==TargettedClass:
                        MalTCP.append(x_test[i][0][0])
                        MalUDP.append(x_test[i][32][0])
                        MalICMP.append(x_test[i][50][0])
                        correct += 1
                        toManipulate.append(x_test[i])
                        if x_test[i][0] > maxMalTCP :
                            maxMalTCP = x_test[i][0]
                            TCPMIndex = i
                        if x_test[i][50] > maxMalICMP :
                            maxMalICMP = x_test[i][50]
                            ICMPMIndex = i
                        if x_test[i][32] > maxMalUDP :
                            maxMalUDP = x_test[i][32]
                            UDPMIndex = i
                if y_testBefore[i] == FromClass and predicts[i] == FromClass:
                    toMask.append(x_test[i])
                    BenignTCP.append(x_test[i][0][0])
                    BenignUDP.append(x_test[i][32][0])
                    BenignICMP.append(x_test[i][50][0])
                    if x_test[i][0] > maxBenignTCP :
                        maxBenignTCP = x_test[i][0]
                        TCPBIndex = i
                    if x_test[i][50] > maxBenignICMP :
                        maxBenignICMP = x_test[i][50]
                        ICMPBIndex = i
                    if x_test[i][32] > maxBenignUDP :
                        maxBenignUDP = x_test[i][32]
                        UDPBIndex = i
            #################### TESTING ##########################
            BenignTCP.sort()
            BenignUDP.sort()
            BenignICMP.sort()
            MalTCP.sort()
            MalUDP.sort()
            MalICMP.sort()
            TCPBMedianIndex = -1
            UDPBMedianIndex = -1
            ICMPBMedianIndex = -1
            TCPMMedianIndex = -1
            UDPMMedianIndex = -1
            ICMPMMedianIndex = -1
            medianTCPB = BenignTCP[int(len(BenignTCP)/2)]
            medianUDPB = BenignUDP[int(len(BenignUDP)/2)]
            medianICMPB = BenignICMP[int(len(BenignICMP)/2)]
            medianTCPM = MalTCP[int(len(MalTCP)/2)]
            medianUDPM = MalUDP[int(len(MalUDP)/2)]
            medianICMPM = MalICMP[int(len(MalICMP)/2)]
            for i in range(len(y_testBefore)):
                if y_testBefore[i] == FromClass and predicts[i] == FromClass:
                    if x_test[i][0][0] == medianTCPB :
                        TCPBMedianIndex = i
                    if x_test[i][32][0] == medianUDPB :
                        UDPBMedianIndex = i
                    if x_test[i][50][0] == medianICMPB :
                        ICMPBMedianIndex = i
                if y_testBefore[i] == 1 and predicts[i] == 1:
                    if x_test[i][0][0] == medianTCPM :
                        TCPMMedianIndex = i
                    if x_test[i][32][0] == medianUDPM :
                        UDPMMedianIndex = i
                    if x_test[i][50][0] == medianICMPM :
                        ICMPMMedianIndex = i

            ############## Median Select ##########################
            # mask = x_test[TCPBMedianIndex]
            # mask = x_test[UDPBMedianIndex]
            # mask = x_test[ICMPBMedianIndex]
            # mask = x_test[TCPMMedianIndex]
            # mask = x_test[UDPMMedianIndex]
            # mask = x_test[ICMPMMedianIndex]
            ############## Max Select ##########################
            # mask = x_test[TCPBIndex]
            # mask = x_test[UDPBIndex]
            # mask = x_test[ICMPBIndex]
            # mask = x_test[TCPMIndex]
            # mask = x_test[UDPMIndex]
            mask = x_test[ICMPMIndex]
            ############## M or B select Select ##########################
            toManipulate = np.asarray(toMask) # Benign
            expected = FromClass
            # toManipulate = np.asarray(toManipulate) # Malicious
            # expected = TargettedClass
            ###################### END #############################
            mask = np.asarray(mask)
            mask = mask.reshape((68))
            toManipulate = toManipulate.reshape((len(toManipulate),68))
            manipulated = []
            for i in range(len(toManipulate)):
                manipulated.append([])
                for j in range(len(toManipulate[i])):
                    if j in toAccumulate :
                        manipulated[i].append(mask[j]+toManipulate[i][j])
                    else:
                        if (j in tcpRelated) and (j in outgoing) :
                            if (toManipulate[i][tcpOut]+mask[tcpOut]) != 0 :
                                manipulated[i].append((toManipulate[i][j]*((toManipulate[i][tcpOut])/(toManipulate[i][tcpOut]+mask[tcpOut]))) + (mask[j]*((mask[tcpOut])/(toManipulate[i][tcpOut]+mask[tcpOut]))))
                            else:
                                manipulated[i].append(0)
                        if (j in tcpRelated) and (j not in outgoing) :
                            if (toManipulate[i][tcpIn]+mask[tcpIn]) != 0 :
                                manipulated[i].append((toManipulate[i][j]*((toManipulate[i][tcpIn])/(toManipulate[i][tcpIn]+mask[tcpIn]))) + (mask[j]*((mask[tcpIn])/(toManipulate[i][tcpIn]+mask[tcpIn]))))
                            else:
                                manipulated[i].append(0)
                        if (j in udpRelated) and (j in outgoing) :
                            if (toManipulate[i][udpOut]+mask[udpOut]) != 0 :
                                manipulated[i].append((toManipulate[i][j]*((toManipulate[i][udpOut])/(toManipulate[i][udpOut]+mask[udpOut]))) + (mask[j]*((mask[udpOut])/(toManipulate[i][udpOut]+mask[udpOut]))))
                            else:
                                manipulated[i].append(0)
                        if (j in udpRelated) and (j not in outgoing) :
                            if (toManipulate[i][udpIn]+mask[udpIn]) != 0 :
                                manipulated[i].append((toManipulate[i][j]*((toManipulate[i][udpIn])/(toManipulate[i][udpIn]+mask[udpIn]))) + (mask[j]*((mask[udpIn])/(toManipulate[i][udpIn]+mask[udpIn]))))
                            else:
                                manipulated[i].append(0)
                        if (j in icmpRelated) and (j in outgoing) :
                            if (toManipulate[i][icmpOut]+mask[icmpOut]) != 0 :
                                manipulated[i].append((toManipulate[i][j]*((toManipulate[i][icmpOut])/(toManipulate[i][icmpOut]+mask[icmpOut]))) + (mask[j]*((mask[icmpOut])/(toManipulate[i][icmpOut]+mask[icmpOut]))))
                            else:
                                manipulated[i].append(0)
                        if (j in icmpRelated) and (j not in outgoing) :
                            if (toManipulate[i][icmpIn]+mask[icmpIn]) != 0 :
                                manipulated[i].append((toManipulate[i][j]*((toManipulate[i][icmpIn])/(toManipulate[i][icmpIn]+mask[icmpIn]))) + (mask[j]*((mask[icmpIn])/(toManipulate[i][icmpIn]+mask[icmpIn]))))
                            else:
                                manipulated[i].append(0)

                totalIncomingPackets = manipulated[i][tcpIn] + manipulated[i][udpIn] + manipulated[i][icmpIn]
                totalOutgoingPackets = manipulated[i][tcpOut] + manipulated[i][udpOut] + manipulated[i][icmpOut]
                totalFractions = []
                if totalIncomingPackets != 0 :
                    totalFractions.append(manipulated[i][tcpIn]/totalIncomingPackets)
                    totalFractions.append(manipulated[i][udpIn]/totalIncomingPackets)
                    totalFractions.append(manipulated[i][icmpIn]/totalIncomingPackets)
                else :
                    totalFractions.append(0)
                    totalFractions.append(0)
                    totalFractions.append(0)
                if totalOutgoingPackets != 0 :
                    totalFractions.append(manipulated[i][tcpOut]/totalOutgoingPackets)
                    totalFractions.append(manipulated[i][udpOut]/totalOutgoingPackets)
                    totalFractions.append(manipulated[i][icmpOut]/totalOutgoingPackets)
                else :
                    totalFractions.append(0)
                    totalFractions.append(0)
                    totalFractions.append(0)
                manipulated[i].append(totalFractions[0])
                manipulated[i].append(totalFractions[3])
                manipulated[i].append(totalFractions[1])
                manipulated[i].append(totalFractions[4])
                manipulated[i].append(totalFractions[2])
                manipulated[i].append(totalFractions[5])
            manipulated = np.asarray(manipulated)
            #print(manipulated.shape)
            # print(manipulated[0])
            # print(toManipulate[0])
            # print(mask)
            manipulated = manipulated.reshape((len(manipulated),68,1))

            predicts = model_argmax(sess,x,preds,manipulated)
            # correct = 0
            # targetCorrect = 0
            # for i in range(len(predicts)):
            #     if predicts[i] == expected :
            #         correct += 1
            #     if predicts[i] == TargettedClass :
            #         targetCorrect+=1
            # print("Misclassification = " + str((1-(correct/len(predicts)))*100))
            # print("Misclassification Targeted= " + str(((targetCorrect/len(predicts)))*100))

            for i in range(len(predicts)):
                confusion[int(predicts[i])][expected]+=1
        co = confusion
        max = [0]*8
        for i in range(len(co)):
            for j in range(len(co[i])):
                max[j] += co[i][j]
        for i in range(len(co)):
            for j in range(len(co[i])):
                co[i][j] = co[i][j]/max[j]
                co[i][j] = round(co[i][j],3)


        co = np.asarray(co)
        df = pd.DataFrame(co, columns=["C0","C1","C2","C3","C4","C5","C6","C7"],index=["C0","C1","C2","C3","C4","C5","C6","C7"])
        sns.set(font_scale=0.925)
        ax = sns.heatmap(df,annot=True,cmap="Blues",fmt='g')
        plt.savefig("/home/ahmed/Documents/sdn/HMaps/ICMP"+str(itarget)+".pdf")
        plt.clf()
        plt.close()
        # for i in range(8):
        #     for j in range(8):
        #         print(confusion[i][j], end ="\t\t")
        #     print("\n")
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
    flags.DEFINE_integer('nb_epochs', 100, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
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
