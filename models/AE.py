import tensorflow as tf
from keras.layers import Dense

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
      self.dense_latent = Dense(self.latent_dim)

    def call(self, inputs):
      x = self.hidden_layers[0](inputs)
      for layer in self.hidden_layers[1:]:
        x = layer(x)
      return self.dense_latent(x)


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

class AE(tf.keras.Model):
  """ Autoencoder """

  def __init__(
    self,
    n_features,
    encoder_neurons,
    decoder_neurons,
    latent_dim,
    hidden_activation,
    output_activation,
    name = 'AE',
    **kwargs
  ):
    super(AE, self).__init__(name=name, **kwargs)
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
    x = self.encoder(inputs)
    reconstructed = self.decoder(x)
    return reconstructed
