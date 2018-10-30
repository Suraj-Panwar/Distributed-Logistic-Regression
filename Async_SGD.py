import argparse
import sys
import numpy as np
import pickle 
import tensorflow as tf

FLAGS = None


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...

	with open('items_train.pkl', 'r') as f:
    		main_data, main_label, index_2_lab, label_list, doc_vec = pickle.load(f)

	print('Checkpoint 1 : Data Loaded\n')

	temp = np.zeros((len(main_label), 50))
	temp[np.arange(len(main_label)), main_label] =1
	main_label = temp

	print('Checkpoint 2 : Data Formatted, Starting Tensorflow Execution\n')

	x = tf.placeholder(tf.float32)
	labels = tf.placeholder(tf.int32)
	W = tf.get_variable('W',[300, 50],initializer=tf.random_normal_initializer(0.,0.3))
	b = tf.get_variable('b',[1,50],initializer=tf.constant_initializer(0.1))

	y_hat = tf.matmul(x, W) + b
	y = tf.nn.softmax(tf.matmul(x, W) + b)

      	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=y_hat))
      	global_step = tf.contrib.framework.get_or_create_global_step()

      	train_op = tf.train.GradientDescentOptimizer(learning rate = 0.01).minimize(
          	loss, global_step=global_step)

    hooks=[tf.train.StopAtStepHook(last_step=1000000)]


    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as sess:
      local_step = 0
      while not sess.should_stop():
	local_step += 1
	for i in range(10000):
		i = np.random.randint(0,main_label.shape[0])
 		if (not sess.should_stop()):
			print(main_label[i,:].reshape(1,50).shape)
			sess.run(train_op, feed_dict= {x:main_data[i,:].reshape(1, 300),labels:main_label[i,:].reshape(1,50)})
		else:
			break
		if (not sess.should_stop()):
			lossy = sess.run(loss)
			print("Loss :", lossy)

		else:
			break

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
