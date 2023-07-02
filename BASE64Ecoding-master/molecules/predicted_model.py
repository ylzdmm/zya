import copy
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, Conv2DTranspose
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D

class VAEencoder_prop():

    encoder_predictor = None
    
    def create(self,
               charset,
               max_length=120,
               latent_rep_size=196,
               weights_file=None,
               encoder_load=None
               ):

        charset_length = charset

        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, z)

        predictor_input = Input(shape=(latent_rep_size,))

        self.predictor = Model(
            predictor_input,
            self._buildPredictor(
                predictor_input
            )
        )

        x1 = Input(shape=(max_length, charset_length))
        loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
        self.encoder_predictor = Model(
            x1,
            self._buildPredictor(
                z1
            )
        )

        if encoder_load:
            self.encoder.load_weights(weights_file, by_name = True)

        if weights_file:
            self.encoder_predictor.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.predictor.load_weights(weights_file, by_name = True)

        self.encoder_predictor.summary()
        self.encoder_predictor.compile(optimizer='Adam',
                                 loss=loss,
                                 metrics=['accuracy'])

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01):
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = 'relu', name='dense_1')(h)

        def sampling(args):   # 采样   std标准差
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]                   #返回张量形状
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)  # 噪声
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        def loss(x, x_prediction):
            mae_loss = objectives.mae(x, x_prediction)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return mae_loss + kl_loss

        return (loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildPredictor(self, z):
        h = Dense(36, name='dense0', activation='tanh')(z)
        h = Dense(28, name='dense1', activation='tanh')(h)
        h = Dense(22, name='dense2', activation='tanh')(h)
        return Dense(1, name='out', activation='linear')(h)

    def save(self, filename):
        self.encoder_predictor.save_weights(filename)
    
    def load(self, charset, length, weights_file, encoder_file, latent_rep_size=196):
        self.create(charset,
                    max_length=length,
                    weights_file=weights_file,
                    encoder_load=encoder_file,
                    latent_rep_size=latent_rep_size
                    )
