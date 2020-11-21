from gcn.layers import *
from gcn.metrics import *
from gcn.render_tf import *
#from gcn.percel_loss import *
from gcn.vgg19 import *
#from gcn.utils_np import render_loss
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, input_name, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = input_name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))
        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_name, input_dim, **kwargs):
        #input_dim = position + feature + color :3 + 23*15 + 3 = 351
        super(GCN, self).__init__(input_name, **kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        #for var in self.layers[0].vars.values():
        #    self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss += tf.nn.l2_loss(self.outputs - self.placeholders['labels'])
        # Cross entropy error
        #self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
        #                                          self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = 0
        #self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
        #                                self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            logging=self.logging))

    def predict(self):
        return self.outputs



class BjyModel(object):
    def __init__(self, input_name, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = input_name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.CNN_layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.paramloss = tf.cast(0, tf.float32)
        self.renderloss = tf.cast(0, tf.float32)
        self.perceploss = tf.cast(0, tf.float32)
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        tf.print(self.inputs)
        for layer in self.CNN_layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.CNN_output = self.activations[-1]
        #转换为feature来执行
        reshape_feature = tf.reshape(self.CNN_output, [128, -1])
        self.activations.append(reshape_feature)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = tf.reshape(self.activations[-1], [128, 3])

        # Store model variables for easy access
        #variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, info, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        #print(self.vars)
        print("in save model !!!!!   ", len(self.vars))
        #result = sess.run(self.vars)
        #np.save("./train_param.npy", result)
        #print(result)
        
        save_path = saver.save(sess, "tmp/%s_test.ckpt" % (self.name))
        print("Model saved in file: %s" % save_path)
        #assert(1==0)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s_test.ckpt" % self.name
        print(save_path)
        saver.restore(sess, save_path)
        
        #result = sess.run(self.vars)
        #np.save("./test_param.npy", result)
        #print(self.vars)
        print(len(self.vars))
        #assert(1==0)
        print("Model restored from file: %s" % save_path)



class BjyGCN(BjyModel):
    def __init__(self, placeholders, input_name, input_dim, **kwargs):
        #input_dim = position + feature + color :3 + 23*15 + 3 = 351
        super(BjyGCN, self).__init__(input_name, **kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        #for var in self.layers[0].vars.values():
        #    self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #self.loss += tf.nn.l2_loss(self.outputs - self.placeholders['labels'])
        #image_pre = render_loss_tensorflow(tf.exp(self.outputs) - 1)
        #image_gt = render_loss_tensorflow(tf.exp(self.placeholders['labels']) - 1)
        image_pre = render_loss_tensorflow(self.outputs)
        image_gt = render_loss_tensorflow(self.placeholders['labels'])
        
        render_loss = image_pre - image_gt
        render_loss = (render_loss ** 2) * 1.0

        #perceptual loss
        input_height = 64
        input_width = 128
        image_pre = tf.reshape(image_pre, (1,input_height, input_width, 3))
        image_gt = tf.reshape(image_gt, (1,input_height, input_width, 3))

        
        percep_loss = perceptual_loss(image_gt, image_pre)
        
        
        sub_result = self.outputs - self.placeholders['labels']
        sub_result = (sub_result ** 2) * 1.0
        self.paramloss = tf.reduce_mean (sub_result)
        self.renderloss = tf.reduce_mean(render_loss)
        self.perceploss = tf.reduce_mean(percep_loss)
        #self.loss = 0.7*tf.reduce_mean (self.placeholders['label_mask'] * sub_result) + 0.3*
        self.loss = self.paramloss + 0.2*self.renderloss + 0.1*self.perceploss
        #self.loss = 0.7*tf.nn.l2_loss (self.outputs - self.placeholders['labels']) + 0.3*percep_loss
        #self.loss = percep_loss
        #self.loss = tf.reduce_mean (self.placeholders['label_mask'] * sub_result)
        #self.loss = tf.reduce_mean (sub_result) 
        # Cross entropy error
        #self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
        #                                          self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = 0
        #self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
        #                                self.placeholders['labels_mask'])

    def _build(self):
        self.CNN_layers.append(tf.layers.Conv2D(filters = 32, kernel_size = 7, padding = 'same', activation=tf.nn.relu))
        self.CNN_layers.append(tf.layers.MaxPooling2D(pool_size = 2, strides = 2))
        self.CNN_layers.append(tf.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation=tf.nn.relu))
        self.CNN_layers.append(tf.layers.MaxPooling2D(pool_size = 2, strides = 2))
        self.CNN_layers.append(tf.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation=tf.nn.relu))
        self.CNN_layers.append(tf.layers.MaxPooling2D(pool_size = 2, strides = 2))
        self.CNN_layers.append(tf.layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation=tf.nn.relu))
        self.CNN_layers.append(tf.layers.MaxPooling2D(pool_size = 2, strides = 2))
        self.CNN_layers.append(tf.layers.Conv2D(filters = 512, kernel_size = 3, padding = 'same', activation=tf.nn.relu))
        self.CNN_layers.append(tf.layers.MaxPooling2D(pool_size = 2, strides = 2))
        self.CNN_layers.append(tf.layers.Flatten())
        self.CNN_layers.append(tf.layers.Dense(units = 384*30, activation=tf.nn.relu))
        self.CNN_layers.append(tf.layers.Dense(units = 10*384, activation=tf.nn.relu))
        #self.layers.append(tf.layers.max_pooling2d(pool_size = 2, strides = 2)
        
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))
        for _ in range(4):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=FLAGS.hidden,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            logging=self.logging))
        #self.layers.append(tf.layers.Dense(units=3))
    def predict(self):
        return self.outputs * 100

