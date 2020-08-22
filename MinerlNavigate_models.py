import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten, Conv2D, add, CuDNNLSTM, TimeDistributed
from keras.losses import mean_squared_error, binary_crossentropy, categorical_crossentropy


class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        # Build models and target models
        self.model = self.network()
        self.target_model = self.network()
        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.model.input[0], self.model.input[1], self.model.input[2]], K.gradients(self.model.output, [self.model.input[2]]))

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        
        action_inputs = Input(shape = (9,), name = 'action_input')
        state = Input(shape = (64, 64, 3), name = 'input_image')
        compassAngle = Input(shape = (1,), name = 'compassAngle')


        x = Conv2D(32, kernel_size = (8,8), strides = (4,4), name = 'layer_1')(state)
        x = Conv2D(64, kernel_size = (4,4), strides = (2,2), name = 'layer_2')(x)
        x = Conv2D(64, kernel_size = (3,3), strides = (1,1), name = 'layer_3')(x)
        x = Flatten()(x)
        x = Dense(256, activation = 'tanh', name = 'dense_1')(x)

        x = concatenate([x, compassAngle])
        x = Dense(128, activation = 'tanh', name = 'dense_2')(x)
        x = concatenate([x, action_inputs])
        x = Dense(128, activation = 'tanh', name = 'dense_3')(x)
        x = Reshape((128,1))(x)
        x = CuDNNLSTM(units = 64, return_sequences=True)(x)
        x = TimeDistributed(Dense(64))(x)
        x = CuDNNLSTM(units= 64, return_sequences=False)(x)
        critic_output = Dense(1, activation = 'linear', name = 'dense_4', kernel_initializer=RandomUniform())(x)

        critic = Model(inputs = [state, compassAngle, action_inputs], outputs = critic_output)
        
        
        return critic

    def gradients(self, state, compassAngle, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        return self.action_grads([state, compassAngle, actions])

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp)

    def train_on_batch(self, state, compassAngle, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([state, compassAngle, actions], critic_target)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')
        self.target_model.save_weights(path + '_critictarget.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
    def load_targetweights(self, path):
        self.target_model.load_weights(path)
     
    

class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau = tau
        self.lr = lr
        self.model = self.network()
        self.target_model = self.network()
        self.adam_optimizer = self.optimizer()
        self.reptile_optimizer = self.pretrain_optimizer()

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        
        state = Input(shape = (64, 64, 3), name = 'input_image')
        compassAngle = Input(shape = (1,), name = 'compassAngle')

        # Actor Model all actions except camera #############################################3
        x = Conv2D(32, kernel_size = (8,8), strides = (4,4), name = 'conv_l1')(state)
        x = Conv2D(64, kernel_size = (4,4), strides = (2,2), name = 'conv_l2')(x)
        x = Conv2D(64, kernel_size = (3,3), strides = (1,1), name = 'conv_l3')(x)
        x = Flatten()(x)
        x = Dense(256, activation = 'tanh', name = 'dense_1')(x)

        x = concatenate([x, compassAngle])

        x = Dense(128, activation = 'tanh', name = 'dense_2')(x)
        x = Reshape((128,1))(x)
        x = CuDNNLSTM(units = 64, return_sequences=True)(x)
        x = TimeDistributed(Dense(64))(x)
        x = CuDNNLSTM(units= 64, return_sequences=False)(x)
        #main_output = Dense(nb_actions, activation = 'tanh', name = 'main_output')(x)

        # 0:Back-Forward, 1:Left-Right, 2:Sneak-Sprint
        movements_action = Dense(3, activation = 'tanh', name = 'movement')(x)
        attack_action = Dense(1, activation = 'sigmoid', name = 'attack')(x)
        jump_action = Dense(1, activation = 'sigmoid', name = 'jump')(x)
        
        #place=[none, dirt]
        place_action = Dense(2, activation = 'softmax', name = 'place_action')(x)
        
        
        #ction_preds = concatenate([movements_action, attack_action, jump_action, place_action])
        ####################################################################################
        
        # For Camera with shared input #####################################################
        x1 = Conv2D(32, kernel_size = (8,8), strides = (4,4), name = 'cam_conv_l1')(state)
        x1 = Conv2D(64, kernel_size = (4,4), strides = (2,2), name = 'cam_conv_l2')(x1)
        x1 = Conv2D(64, kernel_size = (3,3), strides = (1,1), name = 'cam_conv_l3')(x1)
        x1 = Flatten()(x1)
        x1 = Dense(256, activation = 'tanh', name = 'cam_dense_1')(x1)

        x1 = concatenate([x1, compassAngle])

        x1 = Dense(128, activation = 'tanh', name = 'cam_dense_2')(x1)
        x1 = Reshape((128,1))(x1)
        x1 = CuDNNLSTM(units = 64, return_sequences=True)(x1)
        x1 = TimeDistributed(Dense(64))(x1)
        x1 = CuDNNLSTM(units= 64, return_sequences=False)(x1)
        #camera[horizontal, vertical]
        camera_action = Dense(2, activation = 'tanh', name = 'camera')(x1)
        ######################################################################################
        
        
        outputs = concatenate([movements_action, attack_action, jump_action, place_action, camera_action])       

        actor = Model(inputs = [state, compassAngle], outputs = outputs)
        
        return actor

    def predict(self, state):
        """ Action prediction
        """
        #return self.model.predict(np.expand_dims(state, axis=0))
        return self.model.predict(state)

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, state, compassAngle, grads):
        """ Actor Training
        """
        self.adam_optimizer([state,compassAngle, grads])
        
    def pre_train(self, state, compassAngle, actions):
        self.reptile_optimizer([state, compassAngle, actions])
        
    def optimizer(self):
        """ Actor Optimizer
        """
        #print('in optimizer')
        action_gdts = K.placeholder(shape=(None, self.act_dim))
        #print(action_gdts)
        
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function(inputs = [self.model.input[0], self.model.input[1], action_gdts], outputs = [], updates = [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])
    
    def pretrain_optimizer(self):
        """Optimizer for pretraining
        """
        opt = tf.train.AdamOptimizer(self.lr)

        # Compute the gradients for a list of variables.
        action_vec = K.placeholder(shape = (None, self.act_dim))
        #action_preds, camera_preds = self.model.output
        action_preds = self.model.output
        
        loss1 = mean_squared_error(action_vec[:, 0:2] , action_preds[:, 0:2])
        loss2 = binary_crossentropy(action_vec[:, 3], action_preds[:, 3])
        loss3 = binary_crossentropy(action_vec[:, 4], action_preds[:, 4])
        loss4 = categorical_crossentropy(action_vec[:, 5:6], action_preds[:, 5:6])
        loss5 = mean_squared_error(action_vec[:, 7:8] , action_preds[:, 7:8])
        loss = loss1+loss2+loss3+loss4+loss5
        
        grads_and_vars = opt.compute_gradients(loss, self.model.trainable_weights)

        return K.function(inputs = [self.model.input[0], self.model.input[1], action_vec], outputs = [], updates = [opt.apply_gradients(grads_and_vars)])
    

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')
        self.target_model.save_weights(path + '_actortarget.h5')
    def load_weights(self, path):
        self.model.load_weights(path)
    def load_targetweights(self, path):
        self.target_model.load_weights(path)