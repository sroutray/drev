import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as KB
import keras as K
import matplotlib.pyplot as plt

class VAE:

    def __init__(self, input_dim=None, latent_dim=None, intermediate_dim=None, batch_size=None, activation=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.activation = activation
        self.model = None
        self.history = None


    def build(self):
        """ Build the network according to the attributes of the class.
        """

        #### Define encoder ####

        # Define input layer (Type: Tensor)
        input_encoder = Input(batch_shape=(self.batch_size, self.input_dim), name='input_layer_encoder')

        # List of the outputs of the intermediate layers (Type: Tensor)
        output_intermediate_layers_encoder = [None] * len(self.intermediate_dim)

        for n, current_intermediate_dim in enumerate(self.intermediate_dim):
            intermediate_layer_name_encoder = 'intermediate_layer_' + str(n) + '_encoder'
            if n==0:
                # Link to input layer
                output_intermediate_layers_encoder[n] = Dense(current_intermediate_dim,
                                                              activation=self.activation,
                                                              name=intermediate_layer_name_encoder)(input_encoder)
            else:
                # Linked to previous intermediate layer
                output_intermediate_layers_encoder[n] = Dense(current_intermediate_dim,
                                                              activation=self.activation,
                                                              name=intermediate_layer_name_encoder)(output_intermediate_layers_encoder[n-1])

        # Output layer of the encoder: latent variable mean (Type: Tensor)
        latent_mean = Dense(self.latent_dim,
                            name='latent_mean')(output_intermediate_layers_encoder[-1])

        # Output layer of the encoder: logarithm of the latent variable variance (Type: Tensor)
        latent_log_var = Dense(self.latent_dim,
                               name='latent_log_var')(output_intermediate_layers_encoder[-1])

        #### Define sampling layer ####

        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = KB.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,stddev=1.0)
            return z_mean + KB.exp(z_log_var / 2) * epsilon

        # Wrap the sampling function as a layer in Keras (Type: Tensor)
        z = Lambda(sampling)([latent_mean, latent_log_var])

        #### Define decoder ####

        # List of the outputs of the intermediate layers (Type: Tensor)
        output_intermediate_layers_decoder = [None] * len(self.intermediate_dim)

        for n, current_intermediate_dim in enumerate(list(reversed(self.intermediate_dim))):
            intermediate_layer_name_decoder = 'intermediate_layer_' + str(n) + '_decoder'
            if n==0:
                # Link to sampled latent variable layer
                output_intermediate_layers_decoder[n] = Dense(current_intermediate_dim,
                                                              activation=self.activation,
                                                              name=intermediate_layer_name_decoder)(z)
            else:
                # Link to previous intermediate layer
                output_intermediate_layers_decoder[n] = Dense(current_intermediate_dim,
                                                              activation=self.activation,
                                                              name=intermediate_layer_name_decoder)(output_intermediate_layers_decoder[n-1])

        # Output layer of the decoder: log variance of x|z (Type: Tensor)
        output_decoder = Dense(self.input_dim,
                               name='output_layer_decoder') (output_intermediate_layers_decoder[-1])

        self.model = Model(input_encoder, output_decoder)

    def compile(self, optimizer='adam', loss=None):
        """ Compile the model

        Parameters
        ----------
        optimizer   :   string
                        The name of the optimizer as proposed in Keras
        loss        :   string or function
                        The name of objective function as proposed in Keras or an objective function

        Returns
        -------
        Nothing

        """

        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, data_train=None, data_val=None, verbose=1, shuffle=True, epochs=500, early_stopping_dic=None):

        """ Train the model

        Parameters
        ----------
        data_train          :  array, shape (number of examples, dimension)
        data_val            :  array, shape (number of examples, dimension)
        verbose             :  integer
        shuffle             :  boolean
        epochs              :  integer
        early_stopping_dic  :  dictionary


        Returns
        -------
        history             :  History object

        """

        if early_stopping_dic != None:
            # Define callback earlystopping
            early_stopping = K.callbacks.EarlyStopping(monitor=early_stopping_dic['monitor'],
                                                      min_delta=early_stopping_dic['min_delta'],
                                                      patience=early_stopping_dic['patience'],
                                                      verbose=early_stopping_dic['verbose'],
                                                      mode=early_stopping_dic['mode'])
            # Train the VAE
            self.history = self.model.fit(data_train, data_train,
                              verbose=verbose,
                              callbacks=[early_stopping],
                              shuffle=shuffle,
                              epochs=epochs,
                              batch_size=self.batch_size,
                              validation_data=(data_val, data_val))
        else:
            # Train the VAE
            self.history = self.model.fit(data_train, data_train,
                              verbose=verbose,
                              shuffle=shuffle,
                              epochs=epochs,
                              batch_size=self.batch_size,
                              validation_data=(data_val, data_val))

    def print_model(self):
        """ Print the network """
        self.model.summary()

    def save_weights(self, path_to_file):
        """ Save the network weights to a .h5 file specified by 'path_to_file' """
        self.model.save_weights(path_to_file)

    def load_weights(self, path_to_file):
        self.model.load_weights(path_to_file)


    def plot_losses(self):
        """ plot the training and validation losses along the epochs """
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        plt.figure()
        plt.plot(train_loss/np.max(np.abs(train_loss)))
        plt.plot(val_loss/np.max(np.abs(val_loss)))
        plt.title('model normalized loss')
        plt.ylabel('normalized loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def encode_decode(self, data):
        """ Encode the data in the latent space and reconstruct them from the mean of the latent variable """

        # Encode the data in the latent space
        encoder = Encoder(batch_size=self.batch_size)
        encoder.build(self)
        latent_mean, latent_log_var = encoder.encode(data)

        # Decode the data from the mean of the latent variable
        decoder = Decoder()
        decoder.build(self)
        return decoder.decode(latent_mean)


    def encode_decode_with_sampling(self, data):
        """ Encode the data in the latent space and reconstruct them from the mean of the latent variable """

        # Encode the data in the latent space
        encoder = Encoder(batch_size=self.batch_size)
        encoder.build(self)
        latent_mean, latent_log_var = encoder.encode(data)

        latent_sampled = np.zeros((latent_mean.shape[0], latent_mean.shape[1]))
        n_sample = 10
        for n in np.arange(n_sample):
            epsilon = np.random.randn(latent_mean.shape[0], latent_mean.shape[1])
            latent_sampled = latent_sampled + latent_mean +  np.exp(latent_log_var / 2) * epsilon
        latent_sampled = latent_sampled/n_sample

        # Decode the data from the mean of the latent variable
        decoder = Decoder()
        decoder.build(self)
        return decoder.decode(latent_sampled)


class Encoder:
    def __init__(self, batch_size):
        self.model_mean = None
        self.model_log_var = None
        self.batch_size = batch_size

    def build(self, vae):
        self.model_mean = Model(vae.model.get_layer('input_layer_encoder').output, vae.model.get_layer('latent_mean').output)
        self.model_log_var = Model(vae.model.get_layer('input_layer_encoder').output, vae.model.get_layer('latent_log_var').output)

    def encode(self, data):
        latent_mean = self.model_mean.predict(data, batch_size=self.batch_size)
        latent_log_var = self.model_log_var.predict(data, batch_size=self.batch_size)
        return latent_mean, latent_log_var

class Decoder:
    def __init__(self):
        self.model_log_var = None

    def build(self, vae):

        # Input layer with dimension the size of the latent space
        decoder_input = Input(shape=(vae.latent_dim,))

        # List of the intermediate layers (Type: Dense)
        intermediate_layers_decoder = [None] * len(vae.intermediate_dim)

        # List of the outputs of the intermediate layers (Type: Tensor)
        output_intermediate_layers_decoder = [None] * len(vae.intermediate_dim)

        for n, current_intermediate_dim in enumerate(vae.intermediate_dim):
                    intermediate_layer_name_decoder = 'intermediate_layer_' + str(n) + '_decoder'
                    intermediate_layers_decoder[n] = vae.model.get_layer(intermediate_layer_name_decoder)
                    if n==0:
                        # Link to the input of the decoder
                        output_intermediate_layers_decoder[n] = intermediate_layers_decoder[n](decoder_input)
                    else:
                        # Link to previous intermediate layer
                        output_intermediate_layers_decoder[n] = intermediate_layers_decoder[n](output_intermediate_layers_decoder[n-1])

        decoder_output = vae.model.get_layer('output_layer_decoder')(output_intermediate_layers_decoder[-1])

        # Define the decoder model
        self.model_log_var = Model(decoder_input, decoder_output)

    def decode(self, data):
        return self.model_log_var.predict(data)
