#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import pickle
import threading
import queue
import sys
import cv2
import os
import tensorflow as tf
#from skimage import io,transform
kernel_count = 128
input_height = 240
input_width = 360
result_dir = "/root/wcc/indoor_data/train_set/trainset_param/"
ldr_dir = "/root/wcc/indoor_data/train_set/trainset_param/"
data_len = 10000
normal_zone = 5#表示归一化到【-5，5】之间


all_prefix = []
all_files = os.listdir(ldr_dir)
for one in all_files:
    if one.endswith('_sample1.png'):
        all_prefix.append(one[:-12])
print("all prefix len ",len(all_prefix))
#训练集的前缀个数
training_factor = 0.8
data_len = int(np.floor(training_factor * len(all_prefix)))

training_file = []#训练文件
sample_num = 7#采样数量
for i in range(data_len):
	for j in range(sample_num):
		file_path = ldr_dir + all_prefix[i] + "_sample" + str(j) + ".png"
		if os.path.isfile(file_path):
			training_file.append(all_prefix[i] + "_sample" + str(j))
data_len = len(training_file)
print("in training dataset ", len(training_file))
print("for example ", training_file[3], training_file[10])
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def ACESFilm(x):
		a = 2.51
		b = 0.03
		c = 2.43
		d = 0.59
		e = 0.14
		saturate = 1.0
		result = saturate * ((x*(a*x+b))/(x*(c*x+d)+e))
		#print("result", result.shape, result)
		return result
class DataFetcher(threading.Thread):
	def __init__(self, file_list):
		super(DataFetcher, self).__init__()
		self.stopped = False
		self.queue = queue.Queue(64)
		#file_list是读取的文件目录
		self.pkl_list = training_file
		
		self.index = 0
		self.number = data_len
		np.random.shuffle(self.pkl_list)
		print(self.pkl_list[:10])
		

	def work(self, idx):
		pkl_path = self.pkl_list[idx]
		pkl_label = result_dir + pkl_path + "_illu.npy"
		label = np.load(pkl_label)
		#label = np.reshape(label, (128,3))

		img_path = ldr_dir + pkl_path + ".png"

		f = open('actual_list.txt', 'a')
		f.writelines('{}\n'.format(pkl_path))
		
		img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
		
		img = cv2.resize(img, (input_width, input_height))
		img = img / 255.
		
		img = np.expand_dims(img, 0)
		

		#构建mask
		mask = np.ones((label.shape[0], 1))
		'''b_weight = label[:, 0]
		g_weight = label[:, 1]
		r_weight = label[:, 2]
		light_weight = r_weight * 0.3 + g_weight * 0.59 + b_weight * 0.11
		light_weight = (light_weight - np.min(light_weight)) / (np.max(light_weight) - np.min(light_weight))
		light_weight = -normal_zone + 2 * normal_zone * light_weight
		mask = sigmoid(light_weight)
		
		mask = ACESFilm(light_weight)'''
		

		mask = np.reshape(mask, (128,1))
		
		#label = np.log(1 + label)
		#mylist = [0, 5, 8, 13, 16, 18, 21, 26, 29, 34, 39, 42, 47, 50, 55, 60, 63, 68, 71, 73, 76, 81, 84, 89, 94, 97, 102, 105, 110, 115, 118, 123]
		#mask[mylist] = 0
		return img, label, mask
	
	def run(self):
		while self.index < 999999999 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			#if self.index % self.number == 0:
			#		np.random.shuffle(self.pkl_list)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()

if __name__ == '__main__':
	#file_list = sys.argv[1]
	file_list = "as"
	data = DataFetcher(file_list)
	data.start()
	for i in range(99999):
		image,point,mask = data.fetch()
		print("in main " + str(i))
		print("in main image.shape" , image.shape)
		print("in main point.shape" ,point.shape)
		
		print("in main mask.shape" ,mask.shape)
		assert(1==0)
	data.stopped = True
