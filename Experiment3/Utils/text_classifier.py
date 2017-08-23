# This code is mainly borrowed from https://github.com/dennybritz/cnn-text-classification-tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

import tensorflow as tf
import numpy as np

# highway layer that borrowed from https://github.com/carpedm20/lstm-char-cnn-tensorflow
def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = input_
    for idx in xrange(layer_size):
        output = f(tf.nn.rnn_cell._linear(output, size, 0, scope='output_lin_%d' % idx))

        transform_gate = tf.sigmoid(
            tf.nn.rnn_cell._linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
        carry_gate = 1. - transform_gate

        output = transform_gate * output + carry_gate * input_

    return output


class TextLSTM(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, num_hidden, num_layers, pos,BATCH_SIZE,start_token,l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.pos = pos
        self.num_emb = vocab_size
        self.emb_dim = embedding_size
        self.sequence_length = sequence_length
        self.batch_size = BATCH_SIZE
        self.g_params = []
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)

        self.num_classes = num_classes
        self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [self.batch_size, self.num_classes], name="input_y")
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        
        with tf.variable_scope('discriminator'):
            self.g_embeddings = tf.Variable(self.init_matrix_embedding(self.pos))
            self.g_params.append(self.g_embeddings)
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit(self.g_params)  # maps h_t to o_t (output token logits)
 
        # processed for batch
        with tf.device("/cpu:0"):
            inputs = tf.split(1, self.sequence_length, tf.nn.embedding_lookup(self.g_embeddings, self.input_x))
            self.processed_x = tf.pack(
                [tf.squeeze(input_, [1]) for input_ in inputs])  # seq_length x batch_size x emb_dim
        with tf.device("/cpu:0"):
            inputs = tf.split(1, self.sequence_length,  self.input_x)
            self.processed_token_x = tf.pack(
                [tf.squeeze(input_, [1]) for input_ in inputs]) 
        self.h0 = tf.zeros([self.batch_size, self.num_hidden])
        self.h0 = tf.pack([self.h0, self.h0])

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unpack(self.processed_x)
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            print(h_t)
            o_t = self.g_output_unit(h_t)
            print(o_t)
            g_predictions = g_predictions.write(i, o_t)  # batch x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions

        _, _, last, self.g_predictions = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))
        print(last)
        hidden_last, _ = tf.unpack(last)

        def linear(input, output_dim, scope=None, stddev=1.0):
            with tf.variable_scope(scope or 'linear'):
                w = tf.get_variable(
                    'w',
                    [input.get_shape()[1], output_dim],
                    initializer=tf.random_normal_initializer(stddev=stddev)
                )
                b = tf.get_variable(
                    'b',
                    [output_dim],
                    initializer=tf.constant_initializer(0.0)
                )
                return tf.matmul(input, w) + b
        '''
        def minibatch(_input, num_kernels=5, kernel_dim=3):
            x = linear(_input, num_kernels * kernel_dim)
            activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
            diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
            abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
            minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
            return tf.concat(1, [_input, minibatch_features])
        '''
        #min_disc_hidden_last = minibatch(hidden_last)
        min_disc_hidden_last = hidden_last
        print('new layer is here ha ha')
        print(min_disc_hidden_last)
        print('new layer is here waah waah')
        #last = g_predictions.read(self.sequence_length-1) #batch*vocabsize
        #weight, bias = self._weight_and_bias(self.num_hidden + 5, int(self.num_classes))
        weight, bias = self._weight_and_bias(self.num_hidden , int(self.num_classes))
        self.scores = tf.matmul(min_disc_hidden_last, weight) + bias
        self.ypred_for_auc = tf.nn.softmax(self.scores)
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

   
        
     
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) 

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    
    def _weight_and_bias(self,in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def init_matrix_embedding(self,pos):
        if(pos==1):
            data = np.load(open('embedding_pos.npz','rb'))
            word2vecembedding = data['arr_0']
            word2vecembedding = np.array(word2vecembedding,dtype='f')
            print('Loaded pretrained word embeddings')
            print('----------------------------------------------------------------------------')
            
            init = tf.constant(word2vecembedding)
            print(init)
        else:
            data = np.load(open('embedding_neg.npz','rb'))
            word2vecembedding = data['arr_0']
            word2vecembedding = np.array(word2vecembedding,dtype='f')
            print('Loaded pretrained word embeddings')
            print('----------------------------------------------------------------------------')
            
            init = tf.constant(word2vecembedding)
            print(init)

        return init

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.num_hidden]))
        self.Ui = tf.Variable(self.init_matrix([self.emb_dim, self.num_hidden]))
        self.bi = tf.Variable(self.init_matrix([self.num_hidden]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.num_hidden]))
        self.Uf = tf.Variable(self.init_matrix([self.num_hidden, self.num_hidden]))
        self.bf = tf.Variable(self.init_matrix([self.num_hidden]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.num_hidden]))
        self.Uog = tf.Variable(self.init_matrix([self.num_hidden, self.num_hidden]))
        self.bog = tf.Variable(self.init_matrix([self.num_hidden]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.num_hidden]))
        self.Uc = tf.Variable(self.init_matrix([self.num_hidden, self.num_hidden]))
        self.bc = tf.Variable(self.init_matrix([self.num_hidden]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unpack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.pack([current_hidden_state, c])

        return unit
    def init_matrix(self, shape):
        #return tf.random_normal(shape, stddev=0.1)
        '''
        data = np.load(open('embedding.npz','rb'))
        word2vecembedding = data['arr_0']
        word2vecembedding = np.array(word2vecembedding,dtype='f')
        print('Loaded pretrained word embeddings')
        print('----------------------------------------------------------------------------')
        
        init = tf.constant(word2vecembedding)
        print(init)
        '''
        init = tf.random_normal(shape, stddev=0.1)
        print(init)
        return init
    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.num_hidden, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unpack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

   