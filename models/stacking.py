import numpy as np
import tensorflow as tf


class Stacking:

    def __init__(self, args):
        self.num_classes = y_train.shape[1]
        self.num_features = X_train.shape[1]
        self.num_output = y_train.shape[1]
        self.num_layers_0 = 64
        self.num_layers_1 = 128
        self.starter_learning_rate = 0.001
        self.regularizer_rate = 0.1

        # Placeholders for the input data
        input_X = tf.placeholder('float32', shape=(None, self.num_features), name="input_X")
        input_y = tf.placeholder('float32', shape=(None, self.num_classes), name='input_Y')
        ## for dropout layer
        keep_prob = tf.placeholder(tf.float32)

        ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
        self.weights_0 = tf.Variable(
            tf.random_normal([self.num_features, self.num_layers_0], stddev=(1 / tf.sqrt(float(self.num_features)))))
        self.bias_0 = tf.Variable(tf.random_normal([self.num_layers_0]))
        self.weights_1 = tf.Variable(
            tf.random_normal([self.num_layers_0, self.num_layers_1], stddev=(1 / tf.sqrt(float(self.num_layers_0)))))
        self.bias_1 = tf.Variable(tf.random_normal([self.num_layers_1]))
        self.weights_2 = tf.Variable(tf.random_normal([self.num_layers_1, self.num_output], stddev=(1 / tf.sqrt(float(self.num_layers_1)))))
        self.bias_2 = tf.Variable(tf.random_normal([self.num_output]))



    def fit(self, data_x):
        data_x = np.array(data_x)
        train_x = data_x[:, 1:-1]
        train_y = data_x[:, -1]
        input_X = train_x
        input_y = train_y

        hidden_output_0 = tf.nn.relu(tf.matmul(input_X, self.weights_0) + self.bias_0)
        hidden_output_0_0 = tf.nn.dropout(hidden_output_0, self.keep_prob)
        hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0_0, self.weights_1) + self.bias_1)
        hidden_output_1_1 = tf.nn.dropout(hidden_output_1, self.keep_prob)
        predicted_y = tf.sigmoid(tf.matmul(hidden_output_1_1, self.weights_2) + self.bias_2)
        ## Defining the loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y, labels=input_y)) \
                    + self.regularizer_rate * (
                                tf.reduce_sum(tf.square(self.bias_0)) + tf.reduce_sum(tf.square(self.bias_1)))

        ## Variable learning rate
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, 0, 5, 0.85, staircase=True)
        ## Adam optimzer for finding the right weight
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=[self.weights_0, self.weights_1, self.weights_2,
                                                                                   self.bias_0, self.bias_1, self.bias_2])
        ## Metrics definition
        correct_prediction = tf.equal(tf.argmax(y_train, 1), tf.argmax(predicted_y, 1))
        

    def predict(self, test_x):
        test_x = np.array(test_x.iloc[:, 1:], dtype=float)
        pred_y = self.model.predict(test_x)
        return pred_y

