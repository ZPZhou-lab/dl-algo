import tensorflow as tf 
from tensorflow import keras
import numpy as np 
from tensorflow.keras.losses import mean_squared_error, sparse_categorical_crossentropy, binary_crossentropy

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
import seaborn as sns

from typing import Callable, Union, Any, List, Tuple

# load MNIST dataset
def load_mnist_dataset(num_cat : int=0):
    (x_train, y_train), (x_valid, y_valid) = keras.datasets.mnist.load_data()
    # reshape x_train, y_train into a vector
    x_train = np.reshape(x_train, (-1,28*28))
    x_valid = np.reshape(x_valid, (-1,28*28))
    
    # Min-Max Scaler into [0, 1]
    x_train = x_train / 255.0
    x_valid = x_valid / 255.0

    # change dtype into float32 / int32
    x_train, x_valid = np.float32(x_train), np.float32(x_valid)
    y_train, y_valid = np.int32(y_train), np.int32(y_valid)

    # transform the first num_cat columns into categorical features
    bins = [0.2, 0.4, 0.6, 0.8]
    for i in range(num_cat):
        x_train[:,i] = np.digitize(x_train[:,i], bins)
        x_valid[:,i] = np.digitize(x_valid[:,i], bins)
    
    cat_cols = {i:5 for i in range(num_cat)}

    return (x_train, y_train), (x_valid, y_valid), cat_cols

def load_titanic_dataset(valid_size : float=0.3):
    # load titanic dataset
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    feature_cols = ["pclass", "sex", "sibsp", "parch", "embarked", "age", "fare"]
    categorical_cols = ["pclass", "sex", "sibsp", "parch", "embarked"]
    numerical_cols = ["age", "fare"]
    # fetch columns and drop None values
    X = X.loc[:, feature_cols]
    X.dropna(inplace=True, axis=0)
    y = y.loc[X.index].astype(int)

    # Preprocessing
    ordinal_encoder = OrdinalEncoder()
    X.loc[:, categorical_cols] = ordinal_encoder.fit_transform(X.loc[:, categorical_cols])
    scaler = MinMaxScaler()
    X.loc[:, numerical_cols] = scaler.fit_transform(X.loc[:, numerical_cols])

    cat_cols = {i:int(val) for i,val in enumerate(X.loc[:, categorical_cols].max() + 1)}

    # split into train and valid
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, random_state=42)
    # transform into numpy array and change dtype into float32 / int32
    X_train, X_valid = X_train.to_numpy().astype(np.float32), X_valid.to_numpy().astype(np.float32)
    y_train, y_valid = y_train.to_numpy(), y_valid.to_numpy()

    return (X_train, y_train), (X_valid, y_valid), cat_cols

# generate feature matrix mask
def mask_generator(p_m : Union[float, List[float]], X : tf.Tensor) -> tf.Tensor:
    """
    Generate VIME feature matrix mask.

    Parameters
    ----------
    p_m : float or array_like of floats
        The probability to corrupt feature for each column.
    X : tf.Tensor
        The data feature matrix in batch with shape (batch_sz, num_dims).

    Returns
    ----------
    mask : tf.Tensor
        The generated mask matrix with shape (batch_sz, num_dims).
    """

    shape = tf.shape(X)
    mask = keras.backend.random_binomial(shape, p_m, dtype=tf.float32)

    return mask

# generate pretext data
def pretext_generator(mask : tf.Tensor, X : tf.Tensor, num_dims : int) -> Tuple[tf.Tensor]:
    """
    Generate VIME pretext task data.

    Parameters
    ----------
    mask : tf.Tensor
        The generated feature matrix mask with shape (batch_sz, num_dims).
    X : tf.Tensor
        The data feature matrix in batch with shape (batch_sz, num_dims).
    num_dims : int
        The number of features for input data matrix.

    Returns
    ----------
    X_tilde : tf.Tensor
        The generated corrupted data matrix with shape (batch_sz, num_dims).
    mask_tilde : tf.Tensor
        The modified mask matrix with shape (batch_sz, num_dims).
    """

    shape = tf.shape(X)
    n = shape[0]
    # init corrupted feature matrix
    X_bar = tf.zeros_like(X)
    # randomly shuffle data (in column-wise)
    for i in range(num_dims):
        idx = tf.random_shuffle(tf.range(n,dtype=tf.int32))
        X_bar = tf.concat(values=[X_bar[:,:i], tf.gather(X[:,i:i+1], idx), X_bar[:,i+1:]],axis=1)
    
    # corrupt samples
    X_tilde = X * (1 - mask) + X_bar * mask
    # Define new mask matrix
    mask_tilde = tf.cast(tf.not_equal(X, X_tilde), tf.int32)

    return X_tilde, mask_tilde

def fetch_feature_cols(num_dims : int, cat_cols : dict):
    cols = set(list(range(num_dims)))
    cat_idx = set(list(cat_cols.keys()))
    num_idx = cols - cat_idx

    # cat_idx = tf.constant(list(cat_idx), dtype=tf.int32)
    # num_idx = tf.constant(list(num_idx), dtype=tf.int32)
    cat_idx = list(cat_idx)
    num_idx = list(num_idx)

    max_num_cat = max(cat_cols.values()) if len(cat_cols) > 0 else 0

    return num_idx, cat_idx, max_num_cat    

def build_reconstruction_loss(X_unlabel : tf.Tensor, X_num_hat : tf.Tensor, X_cat_hat : tf.Tensor, num_idx : list, cat_idx : list):
    # reconstruction loss for numerical feature
    if X_num_hat is not None:
        X_unlabel_num = tf.gather(X_unlabel, num_idx, axis=1)
        loss_recon_num = tf.reduce_mean(mean_squared_error(X_unlabel_num, X_num_hat))
    else:
        loss_recon_num = 0
    
    # reconstruction loss for categorical feature
    if X_cat_hat is not None:
        X_unlabel_cat = tf.gather(X_unlabel, cat_idx, axis=1)
        loss_recon_cat = tf.reduce_mean(sparse_categorical_crossentropy(X_unlabel_cat, X_cat_hat, from_logits=True))
    else:
        loss_recon_cat = 0
    # merge two loss
    loss_recon = loss_recon_num + loss_recon_cat

    return loss_recon

def build_supervised_loss(y_true : tf.Tensor, y_pred : tf.Tensor, task : str):
    """
    Calculate the supervised loss for VIME model.
    """
    if task == "multiclass":
        loss = tf.reduce_mean(sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))
    elif task == "regression":
        loss = tf.reduce_mean(mean_squared_error(y_true, y_pred))
    elif task == "binary":
        loss = tf.reduce_mean(binary_crossentropy(y_true, y_pred, from_logits=True))
    else:
        raise ValueError("task must be one of 'multiclass', 'regression', 'binary'.")

    return loss

def build_unsupervised_loss(vime_semi : tf.keras.Model, X_unlabel : tf.Tensor, K : int=10, p_m : float=0.2, **kwargs):
    """
    Calculate the unsupervised loss for VIME model.

    Parameters
    ----------
    vime_semi : tf.keras.Model
        The VIME Semi-supervised model.
    X_unlabel : tf.Tensor
        The unlabelled data matrix with shape (batch_sz, num_dims).
    K : int
        The number of augmented samples.
    p_m : float or array_like of floats
        The probability to corrupt feature for each column.
    """

    loss = 0
    # make prediction for original data
    y_hat = vime_semi(X_unlabel, **kwargs)
    
    for _ in range(K):
        # create corrupted data
        mask = mask_generator(p_m, X_unlabel)
        X_tilde, _ = pretext_generator(mask, X_unlabel, vime_semi.num_dims)
        
        # make prediction for augmented corrupted data
        y_hat_aug = vime_semi(X_tilde, **kwargs)

        # calculate loss
        loss += tf.reduce_mean(mean_squared_error(y_hat, y_hat_aug))
    
    return loss / K

def build_selfsupervised_loss(vime_semi : tf.keras.Model, X_unlabel : tf.Tensor, p_m, alpha : float=1.0, **kwargs):
    """
    Calculate the self-supervised loss for VIME model.

    Parameters
    ----------
    vime_semi : tf.keras.Model
        The VIME Semi-supervised model.
    X_unlabel : tf.Tensor
        The unlabelled data matrix with shape (batch_sz, num_dims).
    """

    # create corrupted data
    mask = mask_generator(p_m, X_unlabel)
    X_tilde, mask_tilde = pretext_generator(mask, X_unlabel, vime_semi.num_dims)
    
    # estimate mask and original feature (include numerical & categorical) using corrupted data
    X_num_hat, X_cat_hat, mask_logits = vime_semi(X_tilde, **kwargs)

    # calculate loss
    # part 1: mask estimatation loss
    loss_mask = tf.losses.sigmoid_cross_entropy(mask_tilde, mask_logits)
    # part 2: reconstruction loss
    loss_recon = build_reconstruction_loss(X_unlabel, X_num_hat, X_cat_hat, vime_semi.num_idx, vime_semi.cat_idx)
    loss_total = loss_mask + alpha * loss_recon
    
    return loss_total

def get_latent_representation(X : np.ndarray, estimator : tf.estimator.Estimator):
    # define predict input_fn
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"X_unlabel": X}, shuffle=False)
    
    # PREDICT
    pred_gen = estimator.predict(input_fn=pred_input_fn)

    predictions = {"latent": []}
    for pred in pred_gen:
        predictions["latent"].append(pred["latent"])
    predictions["latent"] = np.vstack(predictions["latent"])

    return predictions["latent"]

def plot_latent_representation(x_latent_train, y_train):
    # decompose integer into two factors
    def decompose_integer(n):
        factors = []
        for i in range(1, int(n ** 0.5) + 1):
            if n % i == 0:
                factors.append(i)
        
        if len(factors) == 0:
            return None
    
        return factors[-1], n // factors[-1]
    
    num_dims = x_latent_train.shape[1] # number of latent dimensions
    # number of subfigures = C(num_dims, 2)
    num_of_subfig = num_dims * (num_dims - 1) // 2 
    w, h = decompose_integer(num_of_subfig)
    num_classes = len(np.unique(y_train))

    fig, ax = plt.subplots(w, h, figsize=(h*3, w*3))
    ax = ax.flatten()
    
    cls_idx = [np.where(y_train == i)[0] for i in range(num_classes)]
    palette = sns.color_palette("muted", num_classes)
    cnt = 0
    for i in range(num_dims):
        for j in range(i+1, num_dims):
            for k in range(num_classes):
                ax[cnt].scatter(x_latent_train[cls_idx[k],i], 
                                x_latent_train[cls_idx[k],j], 
                                label="cls %d"%(k+1), s=5, color=palette[k])
            ax[cnt].set_xlabel("dim %d"%(i))
            ax[cnt].set_ylabel("dim %d"%(j))
            ax[cnt].legend(loc="upper right")
            cnt += 1
    plt.tight_layout()