import tensorflow as tf
from tensorflow import keras

class Sampling(tf.keras.layers.Layer):
    """Sampling de z a partir de z_mean e z_log_var."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.layers.Layer):
    """Implementação de Encoder com layers parametrizáveis"""

    def __init__(
        self,
        neurons,
        latent_dim,
        hidden_activation,
        name = "encoder",
        **kwargs
    ):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.neurons = neurons
        self.latent_dim = latent_dim
        self.hidden_layers = [Dense(neuron, activation=hidden_activation) for neuron in self.neurons]
        self.dense_mean = Dense(self.latent_dim)
        self.dense_log_var = Dense(self.latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.hidden_layers[0](inputs)
        for layer in self.hidden_layers[1:]:
            x = layer(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
    """Implementação de Decoder com layers parametrizáveis"""

    def __init__(
      self,
      n_features,
      neurons,
      hidden_activation,
      output_activation,
      name = "decoder",
      **kwargs
    ):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.neurons = neurons
        self.n_features = n_features
        self.hidden_layers = [Dense(neuron, activation=hidden_activation) for neuron in self.neurons]
        self.dense_output = Dense(self.n_features, activation=output_activation)

    def call(self, inputs):
        x = self.hidden_layers[0](inputs)
        for layer in self.hidden_layers[1:]:
            x = layer(x)
        return self.dense_output(x)

class VAE(tf.keras.Model):
    """ Variational Autoencoder """

    def __init__(
        self,
        n_features,
        encoder_neurons,
        decoder_neurons,
        latent_dim,
        hidden_activation,
        output_activation,
        name = 'VAE',
        **kwargs
    ):
        super(VAE, self).__init__(name=name, **kwargs)
        self.n_features = n_features
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.latent_dim = latent_dim
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.encoder = Encoder(
            latent_dim = self.latent_dim,
            neurons = self.encoder_neurons,
            hidden_activation = self.hidden_activation
        )
        self.decoder = Decoder(
            n_features = self.n_features,
            neurons = self.decoder_neurons,
            hidden_activation = self.hidden_activation,
            output_activation = self.output_activation
        )

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed
