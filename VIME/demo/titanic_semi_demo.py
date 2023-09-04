import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# add parent path to sys.path at runtime
# so that we can import modules under VIME/
import sys
sys.path.append("../")

from estimator import VIMESemiEstimator, VIMESelfEstimator
from model import VIMESelf, Encoder
import utils

if __name__ == "__main__":
    # Load data
    print("load Titanic data...")
    (x_train, y_train), (x_valid, y_valid), cat_cols = utils.load_titanic_dataset()

    # create VIMESelf model
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

    # create Encoder with scope prefix "encoder"
    with tf.variable_scope("encoder"):
        encoder = Encoder(params["num_dims"], num_hiddens=params["latent_sz"], cat_cols=params["cat_cols"], cat_embed_dims=1, dropout=params["dropout"])
        encoder.build(input_shape=(None, params["num_dims"]))
    
    # load encoder checkpoint
    saver = tf.train.Saver(var_list=encoder.variables)
    
    with tf.Session() as sess:
        # restore encoder checkpoint
        saver.restore(sess, "./model_titanic/model.ckpt-300")
        print("encoder restored.")

        # get latent representation
        x_latent_train = sess.run(encoder(x_train))
        x_latent_valid = sess.run(encoder(x_valid))

    # build classifier with latent feature
    print("build classifier with latent feature...")
    model = RandomForestClassifier(n_jobs=-1, min_samples_leaf=16)
    model.fit(x_latent_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(x_latent_train))
    valid_acc = accuracy_score(y_valid, model.predict(x_latent_valid))
    print("train acc using latent feature: %.4f"%(train_acc))
    print("valid acc using latent feature: %.4f"%(valid_acc))


    
    



