import tensorflow as tf 
from tensorflow import keras

from typing import Callable, Union, Any, List, Tuple
from utils import fetch_feature_cols

# define Encoder
class Encoder(tf.keras.Model):
    """
    The Encoder model to encode input data matrix into latent representation, can handle both numerical and categorical features.
    """
    def __init__(self, num_dims : int, latent_sz : int, cat_cols : dict={}, cat_embed_dims : int=1, dropout : float=0.20, **kwargs):
        """
        Parameters
        ----------
        num_dims : int
            The number of features for input data.
        latent_sz : int
            The dimension of latent representation.
        cat_cols : dict, defualt = {}
            The config for categorical columns.
            `key` represents the index for categorical column,
            `value` represents the number of different values for this categorical feature.
        cat_embed_dims : int, default = 1
            The embedding output size for each categorical feature.
        dropout : float, defualt = 0.20
            The dropout rate for encoder.
        """
        super(Encoder, self).__init__(**kwargs)
        # preprocessing
        self.num_idx, self.cat_idx, max_num_cat = fetch_feature_cols(num_dims, cat_cols)
        
        # encoder for categorical features
        self.cat_embed = keras.layers.Embedding(input_dim=max_num_cat, output_dim=cat_embed_dims)
        self.cat_flat = keras.layers.Flatten()

        # encoder for all features
        num_input_dim = len(self.num_idx)
        cat_input_dim = len(self.cat_idx) * cat_embed_dims
        self.fc1 = keras.layers.Dense(2*latent_sz, activation="relu")
        self.fc1.build(input_shape=(None,num_input_dim + cat_input_dim))
        self.dropout = keras.layers.Dropout(rate=dropout)
        self.fc2 = keras.layers.Dense(latent_sz, activation="relu")

        # convert to tf.int32
        self.num_idx = tf.constant(self.num_idx, dtype=tf.int32)
        self.cat_idx = tf.constant(self.cat_idx, dtype=tf.int32)
        
    def call(self, x, **kwargs):
        # categorical features embedding
        x_cat = tf.gather(x, self.cat_idx, axis=1)
        x_cat = self.cat_flat(self.cat_embed(x_cat))

        # numerical features
        x_num = tf.gather(x, self.num_idx, axis=1)
        
        # concat and encode
        z = tf.concat([x_num, x_cat],axis=1)
        z = self.dropout(self.fc1(z), **kwargs)
        z = self.fc2(z)
        
        return z

class VIMESelf(tf.keras.Model):
    """
    The VIME Self-supervised Model.
    """
    def __init__(self, num_dims : int, cat_cols : dict={}, **kwargs):
        super(VIMESelf, self).__init__(**kwargs)
        # preprocessing
        self.num_idx, self.cat_idx, max_num_cat = fetch_feature_cols(num_dims, cat_cols)
        
        # create mask estimator and feature estimator
        self.mask_est = keras.layers.Dense(num_dims)
        
        # numerical feature estimator
        self.num_feat_est = keras.layers.Dense(len(self.num_idx)) if len(self.num_idx) > 0 else None
        # categorical feature estimator
        self.num_of_cat = len(self.cat_idx)
        self.cat_feat_est = keras.layers.Dense(max_num_cat*self.num_of_cat) if self.num_of_cat > 0 else None

        # convert to tf.int32
        self.num_idx = tf.constant(self.num_idx, dtype=tf.int32)
        self.cat_idx = tf.constant(self.cat_idx, dtype=tf.int32)

    def call(self, X_latent, **kwargs):
        """
        X_latent : tf.Tensor
            The latent representation encoded with shape (batch_sz, latent_sz).

        Returns
        ----------
        num_feat_hat : tf.Tensor
            The estimated original data matrix for numerical feature.
        cat_feat_hat : tf.Tensor
            The estimated original data matrix for categorical feature with logits.
        mask_hat : tf.Tensor
            The estimated mask vector.
        """

        # estimate mask vector
        mask_hat = self.mask_est(X_latent)
        
        # estimate original feature matrix
        num_feat_hat = self.num_feat_est(X_latent) if self.num_feat_est else None
        if self.cat_feat_est:
            cat_feat_hat = self.cat_feat_est(X_latent)
            # cat_feat_hat with shape : (batch_sz, num_of_cat, num_each_cat)
            cat_feat_hat = tf.stack(tf.split(cat_feat_hat, self.num_of_cat, axis=1), axis=1)
        else:
            cat_feat_hat = None
        
        return num_feat_hat, cat_feat_hat, mask_hat

class VIMESemi(tf.keras.Model):
    def __init__(self, num_dims : int, output_dims : int,  dropout=0.0, **kwargs):
        super(VIMESemi, self).__init__(**kwargs)        
        # whehter to train VIMESelf model
        self.freeze_vime_self = False
        
        # create predictor
        self.predictor = keras.models.Sequential([
            keras.layers.Dense(4*output_dims, activation="relu"),
            keras.layers.Dropout(rate=dropout),
            keras.layers.Dense(4*output_dims, activation="relu"),
            keras.layers.Dropout(rate=dropout),
            keras.layers.Dense(output_dims)
        ])

        self.num_dims = num_dims
        self.output_dims = output_dims
        self.num_idx = self.vime_self.num_idx
        self.cat_idx = self.vime_self.cat_idx
    
    # make prediction
    def call(self, X_latent : tf.Tensor, **kwargs):
        """
        X_latent : tf.Tensor
            The latent representation encoded with shape (batch_sz, latent_sz).
        """
        
        # make prediction
        y_hat = self.predictor(X_latent, **kwargs)
        return y_hat