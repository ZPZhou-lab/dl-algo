import tensorflow as tf
from module import ViTClassifier
from utils import split_image_into_patches

if __name__ == "__main__":

    # load mnist data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    # normalize data
    x_train, x_test = x_train / 255.0, x_valid / 255.0
    # add a channel dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_valid = x_valid[..., tf.newaxis].astype("float32")

    # extract patches
    x_train_patches = split_image_into_patches(x_train, patch_size=4)
    x_valid_patches = split_image_into_patches(x_valid, patch_size=4)

    # create model
    model = ViTClassifier(
        sequence_length=49,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_classes=10,
        dropout_rate=0.5
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(x_train_patches, y_train, 
        validation_data=(x_valid_patches, y_valid), 
        epochs=10, batch_size=64)