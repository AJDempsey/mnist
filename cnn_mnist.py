from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Application logic here


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # MNIST dataset consists of monochrome images 28x28 images, this is the second parameter,
    # [batch_size, image_height, image_width, colour channels]
    # A batch size of -1 means dynamically compute the size based on the number of input values in features["x"]
    # The features here will be one value for each pixel in each image. The batch size can be tuned as a
    # hyper parameter.
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolution Layer #1
    # This layer generates feature maps from the image, creating 32 channels for the next steps.
    # Apply 32 5x5 filters to the input layer with a ReLU activation.
    # The padding value means that the output will have the same height and width as the input
    # A filter/kernel is a smaller matrix that slides across the input to try and find features.
    # It works by multiplying the input by the kernel and summing all the values in that region.
    # Apply a Rectified Linear Unit function removes all negative numbers from the feature maps.
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # Reduce the size of the input using a max pooling algorithm.
    # Take the max value in the region under a 2x2 matrix reducing the dimensions
    # Stride means how many pixels to move between poolings - here this means move 2x2 pixels.
    # None of the pooling will over lap then. For different size strides use a tuple - [3, 6]
    # This pooling reduces the size of each image by a half.
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolution Layer #2 and Pooling Layer #2
    # Repeat similar steps as above. The output of the max pooling layer is
    # [batch_size, 7, 7, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    # Dense Layer
    # flatten the previous layer 64 filters (channels) of a size 7 by 7 and feed them into a layer of size 1024
    # in the training mode 40% of these results will randomly drop out
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    # Final layer that will show our classification, 0-9, using linear activation
    # The output layer is of a size [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    # The layers are set up to give the general form of the graph and then the training is what we use to
    # find the best matrices and weights to use to get the results we want. We don't need to know what kernels
    # are used at the beginning of the sequence we find the loss values and then work backwards adjusting
    # the matrices values during the training.

    predictions = {
        # Generate class predictions 0-9 based on the highest value in logits (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add softmax_tensor to the graph. It is used for PREDICT and by the
        # logging_hook
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes) using cross entropy (recommended function for
    # multiclass problems)
    # Loss function tells us how closely the model's predictions match the target class
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode) optimizing on the Loss value we found above
    # (Minimize loss)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )

if __name__ == "__main__":
    tf.app.run()
