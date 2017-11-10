#personal best: 53.6424% accurancy

import numpy as np
import tensorflow as tf

def get_data():

    X = np.array([[0,0,0]])
    y = np.array([[0,0,0]])
    input_file = open("tae_dataset.txt","r")

    # there are 4 parameters in the dataset
    # NATIVE SPEAKER: 1 = English Speaker, 2 = non-English speaker
    # SUMMER: 1 = Summer, 2= Regular
    # CLASS SIZE: (numerical)
    # Class attribute: 1 = Low, 2 = Medium, 3 = High
    for line in input_file:
        native_speaker = int(line.split(',')[0])-1
        summer = int(line.split(',')[1])-1
        class_size = 1/(float(line.split(',')[2]))
        pre_eval = int(line.split(',')[3])
        if pre_eval == 1: evaluation = [1,0,0]
        elif pre_eval == 2: evaluation = [0,1,0]
        else: evaluation = [0,0,1]
        X = np.append(X,[[native_speaker,summer,class_size]],axis=0)
        y = np.append(y,[evaluation],axis=0)

    X = np.delete(X, 0, 0)
    y = np.delete(y, 0, 0)

    input_file.close()

    return X,y

#all variables    
x = tf.placeholder(tf.float32, [None, 3])
W1 = tf.Variable(tf.random_uniform([3, 5]))
W2 = tf.Variable(tf.random_uniform([5, 3]))
b1 = tf.Variable(tf.zeros([5]))
b2 = tf.Variable(tf.zeros([3]))

h1 = tf.sigmoid(tf.matmul(x, W1)+b1)
p = tf.matmul(h1, W2) + b2

#the expected output matrix
y = tf.placeholder(tf.float32, [None, 3])

#the learning rate of the Gradient Descent Optimizer
lr = 0.5

cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=p))
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
batch_xs, batch_ys = get_data()
for i in range(50000):
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    if i%5000 == 0:
        # Test trained model
        print("loss = ", sess.run(cross_entropy, feed_dict={x: batch_xs,
                                              y: batch_ys}))
        correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("accurancy = ", sess.run(accuracy, feed_dict={x: batch_xs,
                                                           y: batch_ys}))

