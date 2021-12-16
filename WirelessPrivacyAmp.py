import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy as sp
from numpy import linalg as LA
from scipy.stats import bernoulli

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

trainX = mnist.train.images
trainY = mnist.train.labels
testX = mnist.test.images
testY = mnist.test.labels

size, _ = trainX.shape

idxs = np.random.permutation(size)
randomTrainX = trainX[idxs]
randomTrainY = trainY[idxs]

K = 200
# Group
groupSize = int(np.floor(K/3))
useridxs = np.random.permutation(K)
group = np.zeros(K, dtype=int)
group[useridxs[0:groupSize]] = 1
group[useridxs[groupSize:2*groupSize]] = 2

each = int(np.floor(size/K))

trainX = [[] for i in range(K)]
trainY = [[] for i in range(K)]

for i in range(K):
    trainX[i] = randomTrainX[i*each:(i+1)*each,:]
    trainY[i] = randomTrainY[i*each:(i+1)*each,:]


def nextBatch(data, label, batchSize):
    size, _ = data.shape

    idx = random.sample(range(0, size), batchSize)
    X = data[idx]
    Y = label[idx]

    return X, Y

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

def total_norm(grad):
    total = 0

    for i in range(len(grad)):
        total += np.linalg.norm(grad[i])**2

    return np.sqrt(total)


# Parameters for the classifier
n_input = 784 # Input layer and the length of the image
#n_hidden_1 = 128 # First hidden layer
n_output = 10 # Output layer and the length of the data
d = n_input*n_output + n_output # total number of parameters

# channel parameter
SNR_L = 2
SNR_M = 10
SNR_H = 30
pert_noise_var = 0.1
channel_noise = 1
powert = [d*channel_noise * 10**(SNR_L / 10), d*channel_noise * 10**(SNR_M / 10), d*channel_noise * 10**(SNR_H / 10)]


#L = L_clip # Lipchitz
s = 2*d


# training parameter
learning_rate = 0.001
training_epochs = 4000
batch_size = 128
display_step = 10
MCiteration = 50

trainingAccuracyGlobal = []
LossesGlobal = []
testingAccuracyGlobal = []
epC = []
epL = []
numPart = []
zero_grads = [np.zeros([n_input, n_output]), np.zeros(n_output)]



X = tf.placeholder(tf.float32, shape=[None, n_input])
Y = tf.placeholder(tf.float32, shape=[None, n_output])

weight = {
    'h1': tf.Variable(glorot_init([n_input, n_output]))
}
bias = {
    'b1': tf.Variable(tf.zeros([n_output]))
}


def NNClassifier(x):
    # First layer
    out = tf.add(tf.matmul(x, weight['h1']), bias['b1'])
    return out

logits = NNClassifier(X)
predict_y = tf.nn.softmax(logits)
###### normalize the output

#lossf = tf.reduce_mean(tf.square(predict_y - Y))
lossf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grad_op = optimizer.compute_gradients(lossf, var_list=[weight['h1'], bias['b1']])
grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in grad_op]

train_op = optimizer.apply_gradients(grad_placeholder)

correct_pred = tf.equal(tf.argmax(predict_y, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



for itera in range(1, MCiteration+1):
    trainingAccuracy = []
    Losses = []
    testingAccuracy = []
    ep_C = []
    ep_L = []
    num_Part = []
    weight = {
        'h1': tf.Variable(glorot_init([n_input, n_output]))
    }
    bias = {
        'b1': tf.Variable(tf.zeros([n_output]))
    }

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        kappa = 5
        rho = 0.1
        r = np.zeros(K)
        rtilde = np.zeros(K, dtype=complex)
        g = np.zeros(K)
        L = 0.1

        for epoch in range(1, training_epochs+1):
            for i in range(K):
                rtilde[i] =  complex( (1/np.sqrt(2)) *np.random.normal(0,1), (1/np.sqrt(2)) *np.random.normal(0,1) )

            r = rho*r + (np.sqrt(1 - rho**2) * rtilde)
            g = np.sqrt(kappa/(kappa+1)) + np.sqrt(1/(kappa+1))*r
            h = abs(g)
            
            batch_x = [[] for i in range(K)]
            batch_y = [[] for i in range(K)]

            for i in range(K):
                batch_x[i], batch_y[i] = nextBatch(trainX[i], trainY[i], batch_size)


            # Sampling
            h_th = 2
            prob = np.ones(K)
            for i in range(K):
                prob[i] = min(1, h[i]/h_th)
                
            sampled = np.zeros(K)
            for i in range(K):
                sampled[i] = bernoulli.rvs(prob[i])

            num_Part = np.append(num_Part, sum(sampled))
                
            grads_vars = [[] for i in range(K)]
            for i in range(K):
                if sampled[i] == 1:
                    grads_vars[i] = sess.run(grad_op, feed_dict={X:batch_x[i], Y:batch_y[i]})

            grads = [[[] for j in range(len(zero_grads))] for i in range(K)]

            new_grads = [[] for i in range(len(zero_grads))]
            new_grads_final = [[] for i in range(len(zero_grads))]

            for i in range(K):
                if sampled[i] == 1:
                    for j in range(len(grads_vars[i])):
                        grads[i][j] = grads_vars[i][j][0]

            gradNorm = np.zeros(K)
            for i in range(K):
                if sampled[i] == 1:
                    gradNorm[i] = total_norm(grads[i])

            # Gradient Clipping
            for i in range(K):
                if sampled[i] == 1:
                    for j in range(len(zero_grads)):
                        grads[i][j] = grads[i][j] * min(1, L/gradNorm[i])

            ### alignment parameters
            alpha = np.zeros(K)
            
            for i in range(K):
                if sampled[i] == 1:
                    alpha[i] = min(1 / h[i], np.sqrt(powert[group[i]]) /( d*pert_noise_var + total_norm(grads[i])**2))

            
            
            for j in range(len(zero_grads)):
                new_grads[j] = zero_grads[j]
                if sum(sampled) == 0:
                    new_grads_final[j] = new_grads[j]
                else:
                    noise_shape = zero_grads[j].shape
                    for i in range(K):
                        if sampled[i] == 1:
                        #new_grads[j] = new_grads[j] + C[i] * grads[i][j]
                            r_noise = np.random.normal(0, np.sqrt(pert_noise_var), noise_shape)
                            new_grads[j] = new_grads[j] + sampled[i] * alpha[i] * (grads[i][j] + r_noise)

                    r_noise_channel = np.random.normal(0, np.sqrt(channel_noise), noise_shape)
                    new_grads[j] = new_grads[j] + r_noise_channel

                    new_grads_final[j] = new_grads[j]/(sum(prob))

            
            feed_dict = {}
            for i in range(len(grad_placeholder)):
                feed_dict[grad_placeholder[i][0]] = new_grads_final[i]
            sess.run(train_op, feed_dict=feed_dict)

            # Privacy
            delta_l = 10**(-5)
            delta_p = 10**(-5)
            beta = np.sqrt( np.log(delta_p/2) / (-2*K) )
            kappa2 = (sum(prob) - max(prob)) - beta*K

            ep_l = (2*L)/np.sqrt((1+kappa2)*pert_noise_var + channel_noise) * np.sqrt(2*np.log(1.25/delta_l))
            c = (2*L/np.sqrt(pert_noise_var)) * np.sqrt(2*np.log(1.25/delta_l))
            
            ch = (max(prob)/(1-delta_p)) * (np.exp(c/np.sqrt(sum(prob)-beta*K)) - 1)
            if ch > 1:
                ep_c = np.log(2*max(prob)/(1-delta_p)) + (c/np.sqrt(sum(prob)-beta*K))
            else:
                ep_c = np.log(1 + (max(prob)/(1-delta_p)) * (np.exp(c/np.sqrt(sum(prob)-beta*K)) - 1))

            ep_C = np.append(ep_C, ep_c)
            ep_L = np.append(ep_L, ep_l)


            ###### concatenate for training loss, acc
            
            if epoch % display_step == 0 or epoch == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([lossf, accuracy], feed_dict={X: np.concatenate((batch_x)),
                                                                     Y: np.concatenate((batch_y))})
                print("MCiteration " + str(itera) + ", Step " + str(epoch) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
                testAcc = sess.run(accuracy, feed_dict={X: mnist.test.images,
                                          Y: mnist.test.labels})
                trainingAccuracy = np.append(trainingAccuracy, acc)
                Losses = np.append(Losses, loss)
                testingAccuracy = np.append(testingAccuracy, testAcc)

        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: mnist.test.images,
                                          Y: mnist.test.labels}))
    if trainingAccuracyGlobal == []:
        trainingAccuracyGlobal = trainingAccuracy
    else:
        trainingAccuracyGlobal = trainingAccuracyGlobal + trainingAccuracy
    if LossesGlobal == []:
        LossesGlobal = Losses
    else:
        LossesGlobal += Losses
    if testingAccuracyGlobal == []:
        testingAccuracyGlobal = testingAccuracy
    else:
        testingAccuracyGlobal += testingAccuracy

    if epC == []:
        epC = ep_C
    else:
        epC = epC + ep_C

    if epL == []:
        epL = ep_L
    else:
        epL = epL + ep_L 

    if numPart == []:
        numPart = num_Part
    else:
        numPart = numPart + num_Part

    np.savetxt('loss.csv', LossesGlobal/itera , delimiter=",", fmt="%s")
    np.savetxt('acc.csv', trainingAccuracyGlobal/itera , delimiter=",", fmt="%s")
    np.savetxt('testAcc.csv', testingAccuracyGlobal/itera , delimiter=",", fmt="%s")
    np.savetxt('epC.csv', epC/itera , delimiter=",", fmt="%s")
    np.savetxt('epL.csv', epL/itera , delimiter=",", fmt="%s")
    np.savetxt('numPart.csv', numPart/itera , delimiter=",", fmt="%s")

