import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# add parent path to sys.path at runtime
# so that we can import modules under VIME/
import sys
sys.path.append("../")

from estimator import VIMESelfEstimator
import utils

if __name__ == "__main__":
    # Load data
    print("load Titanic data...")
    (x_train, y_train), (x_valid, y_valid), cat_cols = utils.load_titanic_dataset()

    # define params
    params = {
        "latent_sz": 4,
        "num_dims": 7,
        "alpha": 0.5,
        "learning_rate": 1e-3,
        "dropout": 0.25,
        "p_m": 0.25,
        "cat_cols": cat_cols
    }
    # define train input_fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X_unlabel": x_train},
        y=y_train,
        batch_size=1024,
        num_epochs=None,
        shuffle=True
    )
    # define eval input_fn
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X_unlabel": x_valid},
        num_epochs=1,
        shuffle=False
    )

    # create estimator
    estimator = VIMESelfEstimator(model_dir="./titanic_self_model",params=params)

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
    x_latent_train = utils.get_latent_representation(estimator, x={"X_unlabel": x_train})["latent"]
    x_latent_valid = utils.get_latent_representation(estimator, x={"X_unlabel": x_valid})["latent"]

    # build classifier with latent feature
    print("build classifier with latent feature...")
    model = RandomForestClassifier(n_jobs=-1, min_samples_leaf=16)
    model.fit(x_latent_train, y_train)
    train_auc = roc_auc_score(y_train, model.predict_proba(x_latent_train)[:,1])
    valid_auc = roc_auc_score(y_valid, model.predict_proba(x_latent_valid)[:,1])
    print("train auc using latent feature: %.4f"%(train_auc))
    print("valid auc using latent feature: %.4f"%(valid_auc))

    # plot latent feature
    utils.plot_latent_representation(x_latent_train, y_train, "./latent_representation_self.png")
