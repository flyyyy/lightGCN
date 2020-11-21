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
import tensorflow as tf
def construct_feed_dict(pkl, placeholders):
	"""Construct feed dictionary."""
	coord = pkl[0]
	'''
	pool_idx = pkl[4]
	faces = pkl[5]
	# laplace = pkl[6]
	lape_idx = pkl[7]

	edges = []
	for i in range(1,4):
		adj = pkl[i][1]
		edges.append(adj[0])
	'''
	feed_dict = dict()
	feed_dict.update({placeholders['features']: coord})
	#feed_dict.update({placeholders['edges'][i]: edges[i] for i in range(len(edges))})
	#feed_dict.update({placeholders['faces'][i]: faces[i] for i in range(len(faces))})
	#feed_dict.update({placeholders['pool_idx'][i]: pool_idx[i] for i in range(len(pool_idx))})
	#feed_dict.update({placeholders['lape_idx'][i]: lape_idx[i] for i in range(len(lape_idx))})
	feed_dict.update({placeholders['support'][i]: pkl[1][i] for i in range(len(pkl[1]))})
	feed_dict.update({placeholders['num_features_nonzero']: 3})
	return feed_dict



#这里的render使用np来实现，同时去掉了第一个batch的设定，输入label都是128*3的
import math
resize_width = 128
resize_height = 64
ori_width = resize_width
ori_height = resize_height
kernel_count = 128

def ae_to_xy(azimuth, elevation):
    elevation = np.tanh(elevation) * math.pi/2
    azimuth = np.tanh(azimuth) * math.pi
    y = (azimuth + math.pi) / math.pi / 2 * ori_width
    x = ori_height - 1 - (elevation + math.pi / 2) / math.pi * ori_height
    return x, y
def xy_to_omega(x, y, height, width):
    half_width = width / 2
    half_height = height / 2
    x = (x - half_width + 0.5) / half_width
    y = (y - half_height + 0.5) / half_height
    return (y * np.pi * 0.5, x * np.pi)
data_re_map = np.zeros((resize_width * resize_height, 2), dtype=np.float32)
data_idx = 0
for y in range(resize_height):
    for x in range(resize_width):
        omega = xy_to_omega(x, resize_height - 1 - y, resize_height, resize_width)
        data_re_map[data_idx] = (omega[0], omega[1])
        data_idx += 1
#测试实践
'''
def cal_I(x, alpha, elevation, azimuth, weight):
    #alpha_bound = np.log(0.5)/(np.cos(np.arctan(2.0/np.sqrt(30)))-1)
    elevation = np.tanh(elevation) * math.pi/2
    azimuth = np.tanh(azimuth) * math.pi
    #alpha = torch.relu(alpha - alpha_bound) + alpha_bound
    
    mi = np.sin(x[:,0]).view(-1,1)*np.sin(elevation) + torch.cos(x[:,0]).view(-1,1)*torch.cos(elevation)*torch.cos(x[:,1].view(-1,1).expand(-1,3)-azimuth) - 1
    exp = torch.exp(mi * alpha)
    I = torch.stack((exp, exp, exp), dim=-1) * weight
    I = torch.sum(I, dim=1)
    return I
'''
def cal_I_pre(x, alpha, elevation, azimuth, weight):
    print("begin pre", alpha.shape, elevation.shape, x.shape, weight.shape)
    left = np.sin(elevation) * (np.reshape(np.sin(x[:, :, 0]), [-1, 1]))
    print("lefdt",left.shape)
    right1 = np.tile(np.reshape(x[:, :, 1], [-1, 1]), [1, kernel_count])
    print("right1", right1.shape)
    right1 = np.cos(right1 - azimuth)
    print("right1", right1.shape)
    right2 = np.reshape(np.cos(x[:, :, 0]), [-1, 1]) * np.cos(elevation)
    right = right2 * right1
    print("right2", right2.shape)
    print("right", right.shape)
    mi = left + right - 1
    print("mi ", mi.shape)
    exp = np.exp(mi * alpha)

    weight = np.expand_dims(weight, 1)
    print("exp ", exp.shape)
    print("weight ", weight.shape)
    I = np.stack((exp, exp, exp), axis=-1) * weight
    print("I", I.shape)
    I = np.sum(I, axis=2)
    print("I", I.shape)
    return I
#计算128个光源的方位角和仰角
def get_center_elevation_azimuth(n):
    angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = angle * np.arange(n, dtype=np.float32)
    y = np.linspace(1.0 - 1.0 / n, 1.0 / n - 1.0, n)
    center = np.zeros((n, 2), dtype=np.float32)
    #print(math.asin(y))
    tmp = [math.asin(x) for x in y]
    #print("an", y, tmp)
    center[:, 0] = [math.asin(x) for x in y]
    center[:, 1] = theta
    #print("herh", center[:, 0])
    return center
BATCH_SIZE = 1
pic_alpha = np.zeros((kernel_count), dtype=np.float32)
pic_alpha[:] = math.log(0.5) / (math.cos(math.atan(2.0 / math.sqrt(kernel_count))) - 1.0)
pic_center = get_center_elevation_azimuth(kernel_count)

pic_elevation = pic_center[:,0]
pic_azimuth = pic_center[:, 1]
pic_elevation = np.expand_dims(pic_elevation, 0)
pic_elevation = np.expand_dims(pic_elevation, 0)
pic_elevation = np.reshape(pic_elevation, (1,1,128))
pic_elevation = np.tile(pic_elevation, (BATCH_SIZE, 1,1))

pic_azimuth = np.reshape(pic_azimuth, (1,1,128))
pic_azimuth = np.tile(pic_azimuth, (BATCH_SIZE, 1, 1))
'''
def render_loss(y_pre):
    #y_true是一个图像，y_pre是一个128*3的参数
    alpha = pic_alpha
    center = pic_center
    elevation = pic_elevation
    azimuth = pic_azimuth
    weights = np.reshape(y_pre,  (-1, 128, 3))
    x = np.expand_dims(data_re_map, 0)
    pre_I = cal_I_pre(x, alpha, elevation, azimuth, weights)
    pre_I = np.reshape(pre_I, (-1, resize_height, resize_width, 3))
    return pre_I
'''

def eval(x, weight, alpha, center):
    mi = center[:, 0] * np.cos(x[0]) * np.sin(x[1]) + center[:, 1] * np.sin(x[0]) + -center[:, 2] * np.cos(x[0]) * np.cos(x[1]) - 1.0
    exp = np.exp(alpha * mi)
    I = (weight.T * exp).T
    return np.sum(I, axis=0)
def get_center(n):
    angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = angle * np.arange(n, dtype=np.float32)
    y = np.linspace(1.0 - 1.0 / n, 1.0 / n - 1.0, n)
    r = np.sqrt(1.0 - y * y)
    center = np.zeros((n, 3), dtype=np.float32)
    center[:, 0] = r * np.cos(theta)
    center[:, 1] = y
    center[:, 2] = r * np.sin(theta)
    return center
def render_loss(y_pre):
    alpha = np.zeros((kernel_count), dtype=np.float32)
    alpha[:] = math.log(0.6) / (math.cos(math.atan(2.0 / math.sqrt(kernel_count))) - 1.0)
    center = get_center(kernel_count)
    weight = np.reshape(y_pre,  (128, 3))
    I = np.zeros((resize_height, resize_width, 3), dtype=np.float32)
    for y in range(resize_height):
        for x in range(resize_width):
            omega = xy_to_omega(x, resize_height - 1 - y, resize_height, resize_width)
            I[y, x] = eval(omega, weight, alpha, center)
    return I