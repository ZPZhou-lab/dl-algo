import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

# add parent path to sys.path at runtime
# so that we can import modules under VIME/
import sys
sys.path.append("../")

from estimator import VIMESemiEstimator
import utils

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

if __name__ == "__main__":
    # Load data
    print("load Titanic data...")
    (x_train, y_train), (x_valid, y_valid), cat_cols = utils.load_titanic_dataset()
    # split train dataset into labeled and unlabeled
    x_train_unlabel, x_train_labeled, _, y_train_labeled = train_test_split(x_train, y_train, test_size=0.2)
    print("num of pos samples for labeled data: ", y_train_labeled.sum())
    # pad labeled data with same size as unlabeled data
    num_unlabel = x_train_unlabel.shape[0]
    x_train_labeled, y_train_labeled, labeled_mask = utils.pad_labeled_data(x_train_labeled, y_train_labeled, num_unlabel)

    # create VIMESelf model
    # define params
    params = {
        "num_dims": 7,
        "latent_sz": 4,
        "output_dims": 1,
        "task": "binary",
        "cat_cols": cat_cols,
        
        # or None, then a new encoder will be trained
        "encoder": "./titanic_self_model/model.ckpt-500",
        "freeze_vime_self": False,
        "vime_self_warmup": 0,
        "add_selfsup_loss": True,
        
        "K": 20,
        "learning_rate": 1e-3,
        "dropout": 0.25,
        "p_m": 0.25,
        
        "alpha": 0.5,
        "beta": 1e-2,
        "gamma": 1e-2,
    }

    # define train input_fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X_unlabel": x_train_unlabel, "X_label": x_train_labeled},
        y={"y_label": y_train_labeled, "label_mask": labeled_mask},
        batch_size=512,
        num_epochs=None,
        shuffle=True
    )
    # define eval input_fn
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X_unlabel": x_valid, "X_label": x_valid},
        y={"y_label": y_valid, "label_mask": np.ones(y_valid.shape[0])},
        num_epochs=1,
        shuffle=False
    )

    # create estimator
    estimator = VIMESemiEstimator(model_dir="./titanic_semi_model", params=params)

    # train
    num_epochs = 5
    for i in range(num_epochs):
        # TRAIN
        train_res = estimator.train(input_fn=train_input_fn, steps=100)
        # EVAL
        eval_res = estimator.evaluate(input_fn=eval_input_fn)

    # PREDICT
    # get encoded latent representation
    print("get latent representation...")
    train_pred = utils.get_latent_representation(estimator, x={"X_unlabel": x_train, "X_label": x_train})
    valid_pred = utils.get_latent_representation(estimator, x={"X_unlabel": x_valid, "X_label": x_valid})
    x_latent_train = train_pred["latent"]
    x_latent_valid = valid_pred["latent"]

    # use VIMESemi predictions
    y_train_prob = sigmoid(train_pred["y_hat"]).flatten()
    y_valid_prob = sigmoid(valid_pred["y_hat"]).flatten()

    train_auc = roc_auc_score(y_train, y_train_prob)
    valid_auc = roc_auc_score(y_valid, y_valid_prob)
    print("train auc using VIMESemi: %.4f"%(train_auc))
    print("valid auc using VIMESemi: %.4f"%(valid_auc))

    # build classifier with latent feature
    print("build classifier with latent feature...")
    model = RandomForestClassifier(n_jobs=-1, min_samples_leaf=16)
    model.fit(x_latent_train, y_train)
    train_auc = roc_auc_score(y_train, model.predict_proba(x_latent_train)[:,1])
    valid_auc = roc_auc_score(y_valid, model.predict_proba(x_latent_valid)[:,1])
    print("train auc using latent feature: %.4f"%(train_auc))
    print("valid auc using latent feature: %.4f"%(valid_auc))

    # plot latent feature
    utils.plot_latent_representation(x_latent_train, y_train, "./latent_representation_semi.png")


    
    



