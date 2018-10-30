import numpy as np
import pickle 

with open('items_train.pkl', 'r') as f:
    main_data, main_label, index_2_lab, label_list, doc_vec = pickle.load(f)

print('Checkpoint 1 : Data Loaded\n')

temp = np.zeros((len(main_label), 50))
temp[np.arange(len(main_label)), main_label] =1
main_label = temp

print('Checkpoint 2 : Data Formatted, Starting Tensorflow Execution\n')

import tensorflow as tf
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 300])
labels = tf.placeholder(tf.float32, [None, 50])
W = tf.get_variable('W',[300, 50],initializer=tf.random_normal_initializer(0.,0.3))
b = tf.get_variable('b',[1,50],initializer=tf.constant_initializer(0.1))

y_hat = tf.matmul(x, W) + b
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=y_hat))
train_op = tf.train.GradientDescentOptimizer(0.08).minimize(cross_entropy)
acc =  tf.equal(tf.argmax(y,1), tf.argmax(labels, 1))
per_acc = tf.reduce_mean(tf.cast(acc, tf.float32)) * 100

print('Checkpoint 3 : Model Created, Starting Model Training\n')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10000):
        i = np.random.randint(0,main_label.shape[0])
        sess.run(train_op,feed_dict = {x:main_data[i,:].reshape(1, 300),labels:main_label[i,:].reshape(1,50)})
        if (step % 50000 == 0):
            train_acc = sess.run(per_acc,feed_dict = {x: main_data[0:300],labels: main_label[0:300,:]}) 
            #test_acc = clf.clf_accuracy(mnist.test.images, mnist.test.labels)
            print("Steps {}-> Train Accuracy = {}%".format(step,train_acc))
    weight = sess.run(W)

print('Checkpoint 4 : Model Trained, Computing Accuracy\n')

prediction = np.argmax(np.array(doc_vec.toarray()).dot(weight), 1)
sum1 = 0
for i, j in zip(prediction ,label_list):
	for item in j:
		if i==item:
			sum1 += 1
acc = float(sum1)/float(len(label_list))
print('Training Accuracy of the model : {}% \n'.format(acc*100))

print('Checkpoint 5 : Computed Training Accuracy, Loading Test Data\n')

with open('items_test.pkl', 'rb') as f:
    main_data1, main_label1, label_list, doc_vec = pickle.load(f)

print('Checkpoint 6: Test Data Loaded, Computing Test Accuracy\n')

prediction = np.argmax(np.array(doc_vec.toarray()).dot(weight), 1)
sum1 = 0
for i, j in zip(prediction ,label_list):
	for item in j:
		if i==item:
			sum1 += 1
acc = float(sum1)/float(len(label_list))
print('Test Accuracy of the model : {}% \n'.format(acc*100))

print('Checkpoint 6 : Test Accuracy Computed\n')
