import tensorflow as tf
from typing import Callable, Union, Any, Tuple
import utils
from model import Encoder, VIMESelf, VIMESemi

# Self-supervised Learning Estimator
class VIMESelfEstimator(tf.estimator.Estimator):
    """
    VIME Self-supervised learning estimator.
    """
    def __init__(self, model_dir=None, config=None, params=None, warm_start_from=None):
        """
        Parameters
        ----------
        params : dict
            The parameters for VIMESemi model with following keys:
            num_dims : int
                The number of total features.
            output_dims : int
                The number of output dimensions, i.e. the number of dimensions of label.
            latent_sz : int
                The number of hidden units in encoder.
            cat_cols : dict, default is `{}`
                The categorical columns with column index as key and number of categories as value.
            cat_embed_dims : int, default is `1`
                The number of dimensions for categorical embedding.
            dropout : float, default is `0.0`
                The dropout rate for the model.
            p_m : float, default is `0.2`
                The probability of masking each feature when creating corrupted data.
            alpha : float, default is `1.0`
                The weight for reconstruction loss in self-supervised learning.
            learning_rate : float, default is `0.001`
                The learning rate for training.
        """
        super(VIMESelfEstimator, self).__init__(
            model_fn=self.model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
    
    # define estimator logic
    def model_fn(self, features, labels=None, mode=None, params=None):
        # set learning phase
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.keras.backend.set_learning_phase(True)
        else:
            tf.keras.backend.set_learning_phase(False)
        
        # fetch input data
        X_unlabel = features["X_unlabel"]
        cat_cols = params.get("cat_cols", {})

        # preprocessing cat_cols
        num_idx, cat_idx, _ = utils.fetch_feature_cols(num_dims=params["num_dims"], cat_cols=cat_cols)
        num_idx, cat_idx = tf.constant(num_idx, dtype=tf.int32), tf.constant(cat_idx, dtype=tf.int32)
        
        # define encoder
        with tf.variable_scope("encoder"):
            self.encoder = Encoder(
                num_dims=params["num_dims"], latent_sz=params["latent_sz"], cat_cols=cat_cols, 
                cat_embed_dims=params.get("cat_embed_dims",1), dropout=params.get("dropout", 0.0))
            self.encoder.build(input_shape=(None, params["num_dims"]))
        # define VIMESelf model
        self.vime_self = VIMESelf(num_dims=params["num_dims"], cat_cols=cat_cols)

        # create corrupted data
        mask = utils.mask_generator(params["p_m"], X_unlabel)
        X_tilde, mask_tilde = utils.pretext_generator(mask, X_unlabel, params["num_dims"])
        X_tilde_latent = self.encoder(X_tilde)
        
        # estimate mask and original feature (include numerical & categorical) using corrupted data
        X_num_hat, X_cat_hat, mask_logits = self.vime_self(X_tilde_latent)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "latent": self.encoder(X_unlabel)
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
        """
        Parameters
        ----------
        params : dict
            The parameters for VIMESemi model with following keys:
            num_dims : int
                The number of total features.
            output_dims : int
                The number of output dimensions, i.e. the number of dimensions of label.
            latent_sz : int
                The number of hidden units in encoder.
            task : str, default is `"binary"`
                The task type, `"binary"`, `"multiclass"` of `"regression"` are supported.
            VIMESelf : VIMESelf, default is `None`
                The VIMESelf model. If not given, a new VIMESelf model will be created.
            cat_cols : dict, default is `{}`
                The categorical columns with column index as key and number of categories as value.
            freeze_vime_self : bool, default is `False`
                Whether to freeze VIMESelf model when training.
            vime_self_warmup : int, default is `0`
                The number of epochs for VIMESelf model warmup training.
            cat_embed_dims : int, default is `1`
                The number of dimensions for categorical embedding.
            dropout : float, default is `0.0`
                The dropout rate for the model.
            K : int, default is `10`
                The number of augmented samples for each input data when calculating unsupervised loss.
            p_m : float, default is `0.2`
                The probability of masking each feature when creating corrupted data.
            alpha : float, default is `1.0`
                The weight for reconstruction loss in self-supervised learning.
            beta : float, default is `1.0`
                The weight for unsupervised loss.
            gamma : float, default is `1.0`
                The weight for self-supervised loss in semi-supervised learning.
            learning_rate : float, default is `0.001`
                The learning rate for training.
        """
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
            cat_cols=cat_cols, latent_sz=params["latent_sz"],
            cat_embed_dims=params.get("cat_embed_dims",1), dropout=params.get("dropout", 0.0))
        
        # whether to freeze VIMESelf model
        vime_semi.freeze_vime_self = params.get("freeze_vime_self", False)
        # if vime_self_warmup > 0, train VIMESelf model for several at the beginning
        vime_self_warmup = params.get("vime_self_warmup", 0)
        if vime_self_warmup > 0:
            vime_semi.vime_self.trainable = True

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
        
        # use different training logic according to global step and vime_self_warmup
        global_step = tf.train.get_global_step()
        if global_step is not None and global_step < vime_self_warmup:
            # train VIMESelf model
            loss_total = utils.build_selfsupervised_loss(
                vime_semi.vime_self, X_unlabel, 
                params.get("p_m", 0.2), params.get("alpha", 1.0), training=training)
            
            if mode == tf.estimator.ModeKeys.TRAIN:
                # define optimizer and training operator
                optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
                train_op = optimizer.minimize(loss_total, global_step=tf.train.get_global_step())
            
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, train_op=train_op)
            elif mode == tf.estimator.ModeKeys.EVAL:
                eval_metric_ops = {
                    "loss_self": tf.metrics.mean(loss_total)
                }
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, eval_metric_ops=eval_metric_ops)
        else:
            # control whether to train VIMESelf model after warmup
            vime_semi.vime_self.trainable = not vime_semi.freeze_vime_self

            # calculate loss
            # part 1: supervised loss
            loss_sup = utils.build_supervised_loss(y_label, y_hat, task=task)
            # part 2: unsupervised loss (cosistency regularization)
            loss_unsup = utils.build_unsupervised_loss(
                X_unlabel, vime_semi, 
                params.get("K", 10), params.get("p_m", 0.2), training=training)
            loss_total = loss_sup + params.get("beta", 1.0) * loss_unsup

            # part 3: Self-supervised loss
            if not vime_semi.freeze_vime_self:
                loss_self = utils.build_selfsupervised_loss(
                    vime_semi.vime_self, X_unlabel, 
                    params.get("p_m", 0.2), params.get("alpha", 1.0), training=training)
                loss_total += params.get("gamma", 1.0) * loss_self
            
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
                if not vime_semi.freeze_vime_self:
                    eval_metric_ops["loss_self"] = tf.metrics.mean(loss_self)
                
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, eval_metric_ops=eval_metric_ops)