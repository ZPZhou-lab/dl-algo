import tensorflow as tf
from tensorflow import keras
from typing import Callable, Union, Any, Tuple
import utils
from model import Encoder, VIMESelf, VIMESemi

# Self-supervised Learning Estimator
# Self-supervised Learning Estimator
class VIMESelfEstimator(tf.estimator.Estimator):
    """
    VIME Self-supervised learning estimator.
    """
    def __init__(self, model_dir=None, config=None, params=None, warm_start_from=None):
        super(VIMESelfEstimator, self).__init__(
            model_fn=self.model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
    
    # define estimator logic
    def model_fn(self, features, labels=None, mode=None, params=None):
        # fetch input data
        X_unlabel = features["X_unlabel"]
        cat_cols = params.get("cat_cols", {})

        # preprocessing cat_cols
        num_idx, cat_idx, _ = utils.fetch_feature_cols(num_dims=params["num_dims"], cat_cols=cat_cols)
        num_idx, cat_idx = tf.constant(num_idx, dtype=tf.int32), tf.constant(cat_idx, dtype=tf.int32)
        
        # define encoder
        encoder = Encoder(
            num_dims=params["num_dims"], num_hiddens=params["latent_sz"], cat_cols=cat_cols, 
            cat_embed_dims=params.get("cat_embed_dims",1), dropout=params.get("dropout", 0.0))
        # define VIMESelf model
        vime_self = VIMESelf(encoder=encoder, num_dims=params["num_dims"], cat_cols=cat_cols)
        
        # set training flag
        training = (mode == tf.estimator.ModeKeys.TRAIN)

        # create corrupted data
        mask = utils.mask_generator(params["p_m"], X_unlabel)
        X_tilde, mask_tilde = utils.pretext_generator(mask, X_unlabel, params["num_dims"])
        
        # estimate mask and original feature (include numerical & categorical) using corrupted data
        X_num_hat, X_cat_hat, mask_logits = vime_self(X_tilde, training=training)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "latent": encoder(X_unlabel, training=training)
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # calculate loss
        # part 1: mask estimatation loss
        loss_mask = tf.losses.sigmoid_cross_entropy(mask_tilde, mask_logits)
        # part 2: reconstruction loss
        loss_recon = utils.build_reconstruction_loss(X_unlabel, X_num_hat, X_cat_hat, num_idx, cat_idx)
        loss_total = loss_mask + params["alpha"] * loss_recon

        if mode == tf.estimator.ModeKeys.TRAIN:
            # define optimizer and training operator
            optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
            train_op = optimizer.minimize(loss_total, global_step=tf.train.get_global_step())
            
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, train_op=train_op)
       
        elif mode == tf.estimator.ModeKeys.EVAL:
            mask_hat = tf.cast(tf.greater(mask_logits, tf.constant(0.0)), tf.int32)
            eval_metric_ops = {
                "mask_acc": tf.metrics.accuracy(mask_tilde, mask_hat)
            }
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, eval_metric_ops=eval_metric_ops)

# Semi-supervised Learning Estimator
class VIMESemiEstimator(tf.estimator.Estimator):
    """
    VIME Semi-supervised learning estimator.
    """
    def __init__(self, model_dir=None, config=None, params=None, warm_start_from=None):
        super(VIMESemiEstimator, self).__init__(
            model_fn=self.model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
    
    # define estimator logic
    def model_fn(self, features, labels=None, mode=None, params=None):
        # fetch input data
        X_unlabel = features["X_unlabel"]
        X_label, y_label = features["X_label"], labels["y_label"]
        cat_cols = params.get("cat_cols", {})
        task = params.get("task", "binary")

        # create model
        vime_semi = VIMESemi(
            num_dims=params["num_dims"], output_dims=params["output_dims"], vime_self=params["VIMESelf"],
            cat_cols=cat_cols, num_hiddens=params["latent_sz"],
            cat_embed_dims=params.get("cat_embed_dims",1), dropout=params.get("dropout", 0.0))
        
        # set training flag
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        # make prediction for label data
        z_label = vime_semi.encode(X_label, training=training)
        y_hat = vime_semi(X_label, training=training)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "latent": z_label,
                "y_hat": y_hat
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
        # calculate loss
        # part 1: supervised loss
        loss_sup = utils.build_supervised_loss(y_label, y_hat, task=task)
        # part 2: unsupervised loss (cosistency regularization)
        loss_unsup = utils.build_unsupervised_loss(X_unlabel, vime_semi, params.get("K", 10), params.get("p_m", 0.2))
        loss_total = loss_sup + params["beta"] * loss_unsup

        # part 3: Self-supervised loss
        if vime_semi.freeze_vime_self:
            loss_self = 0 # freeze VIMESelf model and do not train it
        else:
            loss_self = utils.build_selfsupervised_loss(vime_semi, X_unlabel, params.get("p_m", 0.2), params.get("alpha", 1.0))
            loss_total += params["gamma"] * loss_self
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            # define optimizer and training operator
            optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
            train_op = optimizer.minimize(loss_total, global_step=tf.train.get_global_step())
            
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, train_op=train_op)
        
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                "loss_sup": tf.metrics.mean(loss_sup),
                "loss_unsup": tf.metrics.mean(loss_unsup),
                "loss_total": tf.metrics.mean(loss_total)
            }
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, eval_metric_ops=eval_metric_ops)