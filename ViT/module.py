import tensorflow as tf

class LearnablePositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, hidden_size):
        super(LearnablePositionalEmbedding, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(1, self.sequence_length, self.hidden_size),
            initializer=tf.initializers.RandomNormal(),
            trainable=True
        )
        self.built = True

    def call(self, inputs):
        return inputs + self.positional_embedding

    def get_config(self):
        config = super(LearnablePositionalEmbedding, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size
        })
        return config


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 dropout_rate: float = 0.0):
        super(TransformerBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # MultiHead Self Attention
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size,
            dropout=dropout_rate)
        
        # MLP
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation="gelu"),
            tf.keras.layers.Dense(hidden_size)
        ])

        # Layer Normalization
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

        # store attention score
        self._attn_weights = None

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : tf.Tensor
            tensor with shape: `(batch_size, seq_len, hidden_size)`
        """
        x = inputs
        # calculate self-attention
        x_norm = self.ln1(x)
        x_attn, self._attn_weights = self.attn(x_norm, x_norm, return_attention_scores=True)

        # residual connection
        x += x_attn

        # calculate MLP
        x_ffn = self.ffn(self.ln2(x))

        # residual connection
        x += x_ffn

        return x


class VisionTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 sequence_length: int,
                 hidden_size: int,
                 num_layers: int,
                 num_heads: int,
                 dropout_rate: float = 0.1):
        super(VisionTransformerEncoder, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # embedding projection
        self.embedding_projection = tf.keras.layers.Dense(hidden_size)
        # positional embedding
        self.positional_embedding = LearnablePositionalEmbedding(
            sequence_length + 1, hidden_size)

        # cls-head token
        self.cls_token = tf.Variable(tf.random.normal(shape=(1, 1, hidden_size)))

        # transformer layers
        self.transformer_layers = [
            TransformerBlock(hidden_size, num_heads, dropout_rate)
            for _ in range(num_layers)
        ]

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs: tf.Tensor
            input image pathcs with shape `(batch_size, sequence_length, input_size)`
        """

        # do embedding projection
        embed = self.embedding_projection(inputs)
        # concat cls token
        # embed shape: (batch_size, sequence_length + 1, hidden_size)
        bz = tf.shape(embed)[0]
        embed = tf.concat([
            tf.repeat(self.cls_token, repeats=bz, axis=0),
            embed
        ], axis=1)

        # add positional embedding
        embed = self.positional_embedding(embed)

        # transformer layers
        for layer in self.transformer_layers:
            embed = layer(embed)
        
        return embed
    

class VitClassificationHead(tf.keras.layers.Layer):
    def __init__(self, hidden_size: int, num_classes: int):
        super(VitClassificationHead, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.classifier = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(num_classes)
        ])
        # layer normalization
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs: tf.Tensor
            input tensor with shape `(batch_size, sequence_length, hidden_size)`
        """
        # get cls token
        cls_token = inputs[:, 0, :]
        # layer normalization
        logits = self.classifier(self.ln(cls_token))
        return logits
    

class VitClassifier(tf.keras.Model):
    def __init__(self,
                 sequence_length: int,
                 hidden_size: int,
                 num_layers: int,
                 num_heads: int,
                 num_classes: int,
                 dropout_rate: float = 0.1):
        super(VitClassifier, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.encoder = VisionTransformerEncoder(
            sequence_length, hidden_size, num_layers, num_heads, dropout_rate)
        self.classifier = VitClassificationHead(hidden_size, num_classes)

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs: tf.Tensor
            input image pathcs with shape `(batch_size, sequence_length, input_size)`
        """
        # encode
        x = self.encoder(inputs)
        # classify
        logits = self.classifier(x)
        return logits

    def get_config(self):
        config = {
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)