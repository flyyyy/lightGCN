import numpy as np
import os
import cv2
import tensorflow as tf
input_height = 64
input_width = 128
inputShape = (input_height, input_width, 3)
chanDim = -1
load_dir = '/root/LightEstimation/GCN/9_30_summary/CNN4GCN/test_result/CNN8GCN/'
MAX_NUM = 100
#定义输入
input = tf.keras.Input(shape=inputShape)
# 创建特征提取模型
'''h1 = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', 
        input_shape=[input_height, input_width, 3],pooling=None)(input)
model = tf.keras.Model(inputs = input, outputs = h1)
model.summary()
losses = []'''
Vgg = tf.keras.applications.VGG19(include_top=False,weights='imagenet',input_shape=(input_height, input_width,3), input_tensor = input)
Vgg.trainable = False
#Vgg.summary()
Vgg_Pre =tf.keras.Model(inputs =Vgg.input,outputs =[ Vgg.layers[9].output])
model = Vgg_Pre
model.summary()
def Cal_L2(x1, x2):
    result = (x1 - x2) ** 2
    result = np.mean(result)
    return result
'''
for i in range(MAX_NUM):
    gt_file = load_dir + str(i) + "_gt.exr"
    gt_file = cv2.imread(gt_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    gt_file = np.expand_dims(gt_file, 0)
    pre_file = load_dir + str(i) + "_show.exr"
    pre_file = cv2.imread(pre_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    pre_file = np.expand_dims(pre_file, 0)
    gt_feature = model.predict(gt_file)
    pre_feature = model.predict(pre_file)
    loss = Cal_L2(gt_feature, pre_feature)
    losses.append(loss)
'''

def Cal_Percel_Loss(gt_hdr, pre_hdr):
    '''img = np.zeros((1,input_height, input_width,3))
    img = tf.reshape(img , (1,input_height, input_width,3))
    print(img)
    print(model.predict(img,steps=1))'''
    print("pre_hdr", pre_hdr)
    gt = model.predict(gt_hdr,steps=1)
    pre = model.predict(pre_hdr,steps=1)
    loss = Cal_L2(gt, pre)
    print("loss is ",loss)
    return loss

if __name__ == "__main__":
    img = np.zeros((1,input_height, input_width,3))
    img = tf.reshape(img , (1,input_height, input_width,3))
    Cal_Percel_Loss(img, img)

'''
Model: "vgg19"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 64, 128, 3)]      0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 64, 128, 64)       1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 64, 128, 64)       36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 32, 64, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 32, 64, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 32, 64, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 16, 32, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 16, 32, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 16, 32, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 16, 32, 256)       590080    
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 16, 32, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 8, 16, 256)        0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 8, 16, 512)        1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 8, 16, 512)        2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 8, 16, 512)        2359808   
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 8, 16, 512)        2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 4, 8, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 4, 8, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 4, 8, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 4, 8, 512)         2359808   
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 4, 8, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 2, 4, 512)         0         
=================================================================
Total params: 20,024,384
Trainable params: 0
Non-trainable params: 20,024,384
_________________________________________________________________
'''