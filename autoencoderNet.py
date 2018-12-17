import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
import csv
from sklearn.linear_model import LogisticRegression
# from scipy import stats

from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
import pickle

dimension = 64
learning_rate_start = 0.001
batch_size = 128
display_step = 1
examples_to_show = 10
classnum=120
img_size=40
train = pickle.load( open( "data/train.p", "rb" ) )
# train = pickle.load( open( "data/test.p", "rb" ) )
print(train.shape)

test = pickle.load( open( "data/test.p", "rb" ) )
print(test.shape)
train_labels = pickle.load( open( "data/train_label.p", "rb" ) )
# train_labels = pickle.load( open( "data/test_label.p", "rb" ) )
# print(train_labels)
test_labels = pickle.load( open( "data/test_label.p", "rb" ) )

checkpoint_dir = './checkpoint_dir/MyModel'
model_path = './checkpoint_dir/MyModel-50'
file_path = '64d-'
# checkpoint_dir = './checkpoint_dir/MyModel-64d'
# model_path = './checkpoint_dir/MyModel-64d-100'
# file_path = '64d-'
train_images_trans = train/255
test_images_trans = test/255
training_epochs=50
test_example = test_images_trans[0:10]

### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu,name='conv1')
# Now 40x40x16
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same',name='pool1')
# Now 20x20x16
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu,name='conv2')
# Now 20x20x8
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same',name='pool2')
# Now 10x10x8
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu,name='conv3')
# Now 10x10x8
encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same',name='encoded')
# Now 5x5x8
### Decoder
upsample1 = tf.image.resize_images(encoded, size=(10,10), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 10x10x8
conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu,name='conv4')
# Now 10x10x8
upsample2 = tf.image.resize_images(conv4, size=(20,20), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 20x20x8
conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu,name='conv5')
# Now 20x20x8
upsample3 = tf.image.resize_images(conv5, size=(40,40), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 40x40x8
conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu,name='conv6')
# Now 40x40x16
logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=None)
#Now 60x60x3
# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits,name='decoded')

y_true = inputs_
# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
# 定义global_step
global_step = tf.Variable(0, trainable=False)
# 通过指数衰减函数来生成学习率
learning_rate = tf.train.exponential_decay(learning_rate_start, training_epochs, 10, 0.96, staircase=True)

opt = tf.train.AdamOptimizer(learning_rate_start).minimize(cost)

def getBatchdata(data,label,batchsize,ithbatch):
    return data[ithbatch*batchsize: (ithbatch+1)*batchsize], label[ithbatch*batchsize: (ithbatch+1)*batchsize]

training_epochs=100
isTrain=False
train_cost=[]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4


with tf.Session(config=config) as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.save(sess, checkpoint_dir, global_step=training_epochs)
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
    # total_batch = int(mnist.train.num_examples / batch_size)  # 总批数
    if isTrain:
        total_batch = int(len(train_images_trans) / batch_size)  # 总批数
        for epoch in range(training_epochs):
            for i in range(total_batch):
                # batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0

                batch_xs, batch_ys = getBatchdata(train_images_trans,train_labels,batch_size,i)  # max(x) = 1, min(x) = 0
                # Run optimization op (backprop) and cost op (to get loss value)
                batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: batch_xs.reshape((-1, img_size, img_size, 3)),
                                                                 targets_: batch_xs.reshape((-1, img_size, img_size, 3))})
            if epoch % display_step == 0:
                train_cost.append(batch_cost)
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(batch_cost))
        print("Optimization Finished!")


        encode_decode = sess.run(
            decoded, feed_dict={inputs_: test_example.reshape((-1, img_size, img_size, 3))})
#         f, a = plt.subplots(2, 10, figsize=(10, 2))


#         for i in range(10):
#             a[0][i].imshow(test_example[i])
#             a[1][i].imshow(encode_decode[i])
#         plt.savefig(file_path+"testing samples.png")
#         plt.show()

        plt.show()
        plt.plot(train_cost)
        plt.title('training loss curve')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(file_path+'training loss curve.png')
        plt.show()

        # encoder_result = sess.run(encoded, feed_dict={inputs_: test_images.reshape((-1, 28, 28, 1))})
        # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=test_labels)
        # plt.colorbar()
        # plt.title('test data cluster')
        # plt.savefig('test data cluster.png')
        # plt.show()
    else:
        ckpt=ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt:
            saver.restore(sess, model_path)
        else:
            pass

        encodeImg = sess.run(encoded,{inputs_:train_images_trans.reshape((-1, img_size, img_size, 3))})
        encodeImg_test = sess.run(encoded, {inputs_: test_images_trans.reshape((-1, img_size, img_size, 3))})

        print("process train data")
        encodeImg_fla=encodeImg.reshape((-1,200))
        csvFile = open("train_encoded-128d.csv", "w")
        writertrain = csv.writer(csvFile)
        writertrain.writerows(encodeImg_fla)

        print("process test data")

        encodeImg_fla_test = encodeImg_test.reshape((-1, 200))
        csvFile = open("test_encoded-128d.csv", "w")
        writertrain = csv.writer(csvFile)
        writertrain.writerows(encodeImg_fla_test)

if not isTrain:
    from sklearn.ensemble import RandomForestClassifier
    class Random_forest_cf:
        def __init__(self,X,Y,n_estimate=10,depth=10):
            self.rfc = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimate, max_depth = depth)
            self.rfc.fit(X, Y)
        def getscore(self,X,Y):
            return self.rfc.score(X, Y)

    import matplotlib.pyplot as plt
    def randomForestAcc(train,test,train_label,test_label):

        plt.ylabel('Accuracy')
        plt.xlabel('Depth')
        accuracy = []
        max_accuracy=0
        number_of_trees = []
        for i in range(10,100,20):
            rf = Random_forest_cf(train,train_label,400,i)
            _accuracy = rf.getscore(test,test_label)*100
            if(_accuracy>max_accuracy):
                max_accuracy = _accuracy
                _depth = i
            accuracy.append(_accuracy)
            number_of_trees.append(i)
        print("Maximum accuracy achieved is:", max_accuracy," when the depth is:",_depth)
        plt.plot(number_of_trees, accuracy)
    print('Random forest with autoencoder')
    randomForestAcc(encodeImg_fla,encodeImg_fla_test,train_labels,test_labels)
#     print('Random forest without autoencoder')
#     randomForestAcc(train_images, test_images, train_labels, test_labels)
    if not isTrain:
    from sklearn.model_selection import train_test_split
    from sklearn import datasets, svm, metrics
    import datetime as dt
    def SVMAcc(train,test,train_label,test_label):
        param_C = 5
        param_gamma = 0.05
        start_time = dt.datetime.now()
        print('Start learning at {}'.format(str(start_time)))
        svm_without_fe = svm.LinearSVC()
        #classifier = svm.SVC(C=param_C,gamma=param_gamma, decision_function_shape='ovo')

        svm_without_fe.fit(train, train_label)
        end_time = dt.datetime.now()
        print('Stop learning {}'.format(str(end_time)))
        elapsed_time= end_time - start_time
        print('Elapsed learning {}'.format(str(elapsed_time)))

        expected = test_label
        predicted = svm_without_fe.predict(test)
        return metrics.accuracy_score(expected, predicted)
    print("SVM with autoencoder acc: "+str(SVMAcc(encodeImg_fla,encodeImg_fla_test,train_labels,test_labels)))
#     print("SVM without autoencoder acc: "+str(SVMAcc(train_images, test_images, train_labels, test_labels)))



if not isTrain:
    def LRAcc(train,test,train_label,test_label):
        start_time = dt.datetime.now()
        print('Start learning at {}'.format(str(start_time)))
        clf = LogisticRegression()
        #classifier = svm.SVC(C=param_C,gamma=param_gamma, decision_function_shape='ovo')

        clf.fit(train, train_label)
        end_time = dt.datetime.now()
        print('Stop learning {}'.format(str(end_time)))
        elapsed_time= end_time - start_time
        print('Elapsed learning {}'.format(str(elapsed_time)))

        expected = test_label
        predicted = clf.predict(test)
        return metrics.accuracy_score(expected, predicted)
    print("Logistic Regression with autoencoder acc: "+ str(LRAcc(encodeImg_fla,encodeImg_fla_test,train_labels,test_labels)))
