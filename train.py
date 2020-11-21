import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
from gcn.utils import *
from gcn.models import BjyGCN
from gcn.fetcher import *

import math


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time 
# time.sleep(8800)
# Set random seed
seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_list', 'Data/train_list.txt', 'Data list.') # training data list
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')#学习率
flags.DEFINE_integer('epochs',20, 'Number of epochs to train.')#训练次数
flags.DEFINE_integer('hidden', 128, 'Number of units in hidden layer.') # gcn hidden layer channel中间的维度
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.') # image feature dim
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.') 
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')

# Define placeholders(dict) and model
num_blocks = 3
num_supports = 2
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 240, 360, 3)),  # 输入的feature数值
    'img_inp': tf.placeholder(tf.float32, shape=(None, 240, 360, 3)),  # 输入的图像信息
    'labels': tf.placeholder(tf.float32, shape=(None, 3)),  # GT的光源信息
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],  # 邻接表
    'num_features_nonzero': tf.placeholder(tf.float32),
    'label_mask': tf.placeholder(tf.float32, shape=(None, 1))  # 用来筛选强度比较大的label
}

# define model name and png name
# 需要修改model dat文件，训练文件的命名，图像的命名

model_name = "nomask_01perceploss_02render_CNNDense4GCN_LR5_E5"  # "weightsigmoidMask5_CNNDense4GCN_LR6_E100"
model = BjyGCN(placeholders, input_name=model_name, input_dim=30, logging=True)

# Load data, initialize session
data = DataFetcher(FLAGS.data_list)
data.setDaemon(True)
data.start()
config=tf.ConfigProto()
#config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

# Train graph model
train_loss = open("record_train_loss_continue_237_" + str(FLAGS.epochs) + ".txt", 'a')
train_loss.write('Start training %s, lr =  %f, Epoch = %d ,hidden = %d\n'%(model_name, FLAGS.learning_rate, FLAGS.epochs, FLAGS.hidden))
#pkl 邻接关系
pkl = pickle.load(open('./data_helper/bjy_Edge10.dat', 'rb'))
feed_dict = construct_feed_dict(pkl, placeholders)
#print("place holder ", placeholders)

#训练的数据量
model.load(sess)
train_number = data.number
my_all_loss = np.zeros(FLAGS.epochs)
my_param_loss = np.zeros(FLAGS.epochs)
my_render_loss = np.zeros(FLAGS.epochs)
my_percep_loss = np.zeros(FLAGS.epochs)

tf.add_to_collection('network-output', model.outputs)
for epoch in range(FLAGS.epochs):
	all_loss = np.zeros(train_number,dtype='float32') 
	all_param_loss = np.zeros(train_number,dtype='float32') 
	all_render_loss = np.zeros(train_number,dtype='float32') 
	all_percep_loss = np.zeros(train_number,dtype='float32') 
	print("begin epoch" + str(epoch))
	for iters in range(train_number):
		# Fetch training data
		img_feature, y_train, according_label_mask = data.fetch()
		
		feed_dict.update({placeholders['features']: img_feature})
		feed_dict.update({placeholders['labels']: y_train})
		feed_dict.update({placeholders['label_mask']:according_label_mask})

		# Training step
		#vert = sess.run(model.outputs, feed_dict=feed_dict)
		#vert = sess.run(model.outputs, feed_dict=feed_dict)
		_, modelloss,paramloss,renderloss,perceploss,out1 = sess.run([model.opt_op,model.loss,model.paramloss, model.renderloss, model.perceploss, model.outputs], feed_dict=feed_dict)
		#print("in 85line ", out1)
		'''if iters == 1:
			true_image = render_loss(out1)
			pre_image = render_loss(vert)
			print('here  Epoch %d, Iteration %d'%(epoch,iters))
			save_path = "/home/bjy/LightEstimation/GCN/9_22/test_result/train_result/" + str(epoch) + "_gt.exr"
			cv2.imwrite(save_path, true_image[0].eval(session=sess))
			save_path = "/home/bjy/LightEstimation/GCN/9_22/test_result/train_result/" + str(epoch) + "_test_show.exr"
			cv2.imwrite(save_path, pre_image[0].eval(session=sess))
			#assert(1==0)'''
		#存loss
		all_loss[iters] = modelloss
		all_param_loss[iters] = paramloss
		all_render_loss[iters] = renderloss
		all_percep_loss[iters] = perceploss
		
		mean_loss = np.mean(all_loss[np.where(all_loss)])
		mean_paramloss = np.mean(all_param_loss[np.where(all_param_loss)])
		mean_renderloss = np.mean(all_render_loss[np.where(all_render_loss)])
		mean_perceploss = np.mean(all_render_loss[np.where(all_percep_loss)])
		if (iters+1) % 2000 == 0:
			#print("in 101line ", all_loss)
			print('Epoch %d, Iteration %d'%(epoch + 1,iters + 1))
			print('Mean loss = %f, param loss = %f, render loss = %f, percep loss = %f, iter loss = %f, %d'%(
					mean_loss,paramloss, renderloss, perceploss, modelloss, data.queue.qsize()))
	# Save model
	#model_path = "./model/" + model_name + "train.ckpt"
	#saver = tf.train.Saver()
	#saver.save(sess, model_path)
	model.save(epoch, sess)
	mean_loss = np.mean(all_loss)
	my_all_loss[epoch] = mean_loss
	mean_param_loss = np.mean(all_param_loss)
	my_param_loss[epoch] = mean_param_loss
	mean_render_loss = np.mean(all_render_loss)
	my_render_loss[epoch] = mean_render_loss
	mean_percep_loss = np.mean(all_percep_loss)
	my_percep_loss[epoch] = mean_percep_loss
	train_loss.write('Epoch %d, loss %f, param loss %f render loss %f percep loss %f\n'%(
					epoch+1, mean_loss, mean_param_loss,mean_render_loss, mean_percep_loss))
	
	train_loss.flush()
#保存文件
import matplotlib.pyplot as plt
x = np.arange(1,FLAGS.epochs+1,1)
plt.plot(x, my_all_loss, color='red', label='all loss')
plt.plot(x, my_param_loss, color='green', label='param loss')
plt.plot(x, my_render_loss, color='blue', label='render loss')
plt.plot(x, my_percep_loss, color='black', label='percep loss')
# plt.plot(history.history['val_accuracy'])
plt.legend()
plt.savefig("loss_" + model_name + "_continue237_" + str(FLAGS.epochs) + ".png")

data.shutdown()
train_loss.close()
print('Training Finished!')
