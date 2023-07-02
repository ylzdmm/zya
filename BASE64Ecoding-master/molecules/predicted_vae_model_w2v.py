import copy
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, Conv2DTranspose
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D

class VAE_prop():

    encoder_predictor = None
    
    def create(self,
               charset,
               max_length=120,
               latent_rep_size=196,
               weights_file=None,
               ):
        charset_length = charset
        epsilon_std = 1
        def sampling(args):   # 采样   std标准差
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]                   #返回张量形状
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)  # 噪声
            #return z_mean_ + K.exp(z_log_var_ / 2) * epsilon
            return z_mean_

        x = Input(shape=(max_length, charset_length))
        z_mean, z_log_var = self._buildEncoder(x, latent_rep_size)
        z = Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])

        self.encoder = Model(x, z)

        predictor_input = Input(shape=(latent_rep_size,))

        self.decoder = Model(
            predictor_input,
            self._buildDecoder(
                predictor_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        self.predictor = Model(
            predictor_input,
            self._buildPredictor(
                predictor_input
            )
        )


        self.vae_predictor = Model(
            x,
            [self._buildDecoder(z,
                latent_rep_size,
                max_length,
                charset_length),
             self._buildPredictor(
                 z
             )]
        )

        def vae_loss(x, decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(decoded_mean)
            xent_loss = max_length * objectives.mse(x, x_decoded_mean)   # 重构loss,
            #xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)   # 重构loss,
            # binary_crossentropy是对数误差
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)   # 求kl散度，即kl loss
            # K.mean求均值。K.square求平方
            return xent_loss + kl_loss


        if weights_file:
            self.vae_predictor.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name=True)
            self.predictor.load_weights(weights_file, by_name = True)

        self.vae_predictor.summary()
        loss = {}
        loss['decoded_mean'] = vae_loss
        loss['out'] = 'mae'
        loss_weights = {}
        loss_weights['decoded_mean'] = K.variable(1.0)
        loss_weights['out'] = K.variable(0.5)
        self.vae_predictor.compile(optimizer='Adam',
                                 loss=loss,
                                 loss_weights=loss_weights,
                                 metrics={'decoded_mean': 'accuracy',
                                          'out': 'accuracy'})

    def _buildEncoder(self, x, latent_rep_size):
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = 'relu', name='dense_1')(h)

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        return (z_mean, z_log_var)

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)                     #重复输入
        h = GRU(488, return_sequences=True, name='gru_1')(h)
        h = GRU(488, return_sequences=True, name='gru_2')(h)
        h = GRU(488, return_sequences=True, name='gru_3')(h)
        return TimeDistributed(Dense(charset_length, activation='linear'), name='decoded_mean')(h)

    def _buildPredictor(self, z):
        h = Dense(36, name='dense0', activation='tanh')(z)
        h = Dense(28, name='dense1', activation='tanh')(h)
        h = Dense(22, name='dense2', activation='tanh')(h)
        return Dense(1, name='out', activation='linear')(h)

    def save(self, filename):
        self.vae_predictor.save_weights(filename)
    
    def load(self, charset, length, weights_file, latent_rep_size=196):
        self.create(charset,
                    max_length=length,
                    weights_file=weights_file,
                    latent_rep_size=latent_rep_size
                    )
