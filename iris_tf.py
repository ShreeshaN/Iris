import tensorflow as tf
from src.iris import data_loader
import numpy as np


class Network:
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)



if(__name__=='__main__'):
    net = Network()
    sess = tf.InteractiveSession()
    filepath = 'Iris.csv'
    train_in, train_out,test_in,test_out = data_loader.loadNclean_dataForTf(filepath)


    # Input and output layer placeholders
    inp = tf.placeholder(tf.float32,[None,4])
    oup = tf.placeholder(tf.float32,[None,3])

    # Weights and biases
    weights = tf.Variable(tf.random_normal([4,3],seed=1))
    biases = tf.Variable(tf.random_normal([3],seed=1))
    sess.run(tf.initialize_all_variables())

    # Activation
    activations = tf.nn.sigmoid(tf.matmul(inp,weights)+biases)

    # Cost
    cross_entropy = tf.reduce_mean(((oup * tf.log(activations))+(oup-1*tf.log(activations-1)))* -1)

    train_Step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)




    for i in range(500):

        # Train the model with the training data
        train_Step.run(feed_dict={inp:train_in,oup:train_out})

        # Test the model by giving test data after each train
        pred = sess.run(activations,feed_dict={inp:test_in})

        # Manual accuracy calculation
        accuracy = [(np.argmax(x),np.argmax(y)) for x,y in zip(pred,test_out)]
        result = (sum(int(x==y) for (x,y) in accuracy))/len(test_in)*100
        print ("Epoch:{0} --> {1}%".format(i,result))


        # tf way of calculating accuracy
        # pred = tf.equal(tf.argmax(oup, 1), tf.argmax(activations, 1))
        # accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
        # print(sess.run(accuracy, feed_dict={inp: test_in, oup: test_out}))
