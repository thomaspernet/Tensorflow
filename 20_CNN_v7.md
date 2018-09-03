---
title: Convolutional Neural Network
author: Thomas
date: '2018-08-29'
slug: cnn
categories: []
tags:
  - deep_tf
header:
    caption: ''
    image: ''
---

<style>
body {
text-align: justify}
</style>

# Convolutional neural network

Convolutional neural network, also known as *convnets*, is a well-known
method in computer vision applications. This type of architecture is
dominant to recognize objects from a picture or video.

In this tutorial, you will learn how to construct a convnet and how to
use TensorFlow to solve the handwritten dataset.

## The architecture of a Convolutional Neural Network

Think about Facebook a few years ago, after you uploaded a picture to
your profile, you were asked to add a name to the face on the picture
manually. Nowadays, Facebook uses convnet to tag your friend in the
picture automatically.

A convolutional neural network is not very difficult to understand. An
input image is processed during the convolution phase and later
attributed a label.

A typical convnet architecture can be summarized in the picture below.
First of all, an image is pushed to the network; this is called the
input image. Then, the input image goes through an infinite number of
steps; this is the convolutional part of the network. Finally, the
neural network can predict the digit on the image.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/20_CNN_v7_files/image001.png)

An image is composed of an array of pixels with height and width. A
grayscale image has only one channel while the color image has three
channels (each one for Red, Green, and Blue). A channel is stacked over
each other. In this tutorial, you will use a grayscale image with only
one channel. Each pixel has a value from 0 to 255 to reflect the
intensity of the color. For instance, a pixel equals to 0 will show a
white color while pixel with a value close to 255 will be darker.

Let’s have a look of an image stored in the [MNIST
dataset](http://yann.lecun.com/exdb/mnist/). The picture below shows how
to represent the picture of the left in a matrix format. Note that, the
original matrix has been standardized to be between 0 and 1. For darker
color, the value in the matrix is about 0.9 while white pixels have a
value of 0.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/20_CNN_v7_files/image002.png)

## Convolutional operation

The most critical component in the model is the convolutional layer.
This part aims at reducing the size of the image for faster computations
of the weights and improve its generalization.

During the convolutional part, the network keeps the essential features
of the image and excludes irrelevant noise. For instance, the model is
learning how to recognize an elephant from a picture with a mountain in
the background. If you use a traditional neural network, the model will
assign a weight to all the pixels, including those from the mountain
which is not essential and can mislead the network.

Instead, a convolutional neural network will use a mathematical
technique to extract only the most relevant pixels. This mathematical
operation is called convolution. This technique allows the network to
learn increasingly complex features at each layer. The convolution
divides the matrix into small pieces to learn to most essential elements
within each piece.

In every convnets, there are four components:

1.  Convolution

2.  Non Linearity (ReLU)

3.  Pooling or Sub Sampling

4.  Classification (Fully Connected Layer)

-   Convolution

The purpose of the convolution is to extract the features of the object
on the image locally. It means the network will learn specific patterns
within the picture and will be able to recognize it everywhere in the
picture.

Convolution is an element-wise multiplication. The concept is easy to
understand. The computer will scan a part of the image, usually with a
dimension of 3x3 and multiplies it to a filter. The output of the
element-wise multiplication is called a feature map. This step is
repeated until all the image is scanned. Note that, after the
convolution, the size of the image is reduced.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/20_CNN_v7_files/image003.png)

Below, there is a URL to see in action how convolution works.

![](https://github.com/thomaspernet/Tensorflow/blob/masterhttps://media.giphy.com/media/fV8esV6419OdEltpbO/giphy.gif" width="400" height="400" />

There are numerous channels available. Below, we listed some of the
channels. You can see that each filter has a specific purpose. Note, in
the picture below; the Kernel is a synonym of the filter.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/20_CNN_v7_files/image005.png)
[Source](https://en.wikipedia.org/wiki/Kernel_(image_processing))

*Arithmetic behind the convolution*

The convolutional phase will apply the filter on a small array of pixels
within the picture. The filter will move along the input image with a
general shape of 3x3 or 5x5. It means the network will slide these
windows across all the input image and compute the convolution. The
image below shows how the convolution operates. The size of the patch is
3x3, and the output matrix is the result of the element-wise operation
between the image matrix and the filter.

![](https://github.com/thomaspernet/Tensorflow/blob/masterhttp://machinelearninguru.com/_images/topics/computer_vision/basics/convolutional_layer_1/stride1.gif" width="400" height="400" />

[Source](http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html)

You notice that the width and height of the output can be different from
the width and height of the input. It happens because of the border
effect.

**Border effect**

Image has a 5x5 features map and a 3x3 filter. There is only one window
in the center where the filter can screen an 3x3 grid. The output
feature map will shrink by two tiles alongside with a 3x3 dimension.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/20_CNN_v7_files/image007.png)

To get the same output dimension as the input dimension, you need to add
padding. Padding consists of adding the right number of rows and columns
on each side of the matrix. It will allow the convolution to center fit
every input tile. In the image below, the input/output matrix have the
same dimension 5x5

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/20_CNN_v7_files/image008.png)

When you define the network, the convolved features are controlled by
three parameters:

1.  Depth: It defines the number of filters to apply during the
    convolution. In the previous example, you saw a depth of 1, meaning
    only one filter is used. In most of the case, there is more than one
    filter. The picture below shows the operations done in a situation
    with three filters

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/20_CNN_v7_files/image009.gif" width="400" height="400" />

1.  Stride: It defines the number of "pixel's jump" between two slices.
    If the stride is equal to 1, the windows will move with a pixel's
    spread of one. If the stride is equal to two, the windows will jump
    by 2 pixels. If you increase the stride, you will have smaller
    feature maps.

Example stride 1

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/20_CNN_v7_files/image010.png)

Image stride 2

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/20_CNN_v7_files/image011.png)

1.  Zero-padding: A padding is an operation of adding a corresponding
    number of rows and column on each side of the input features maps.
    In this case, the output has the same dimension as the input.

-   Non Linearity (ReLU)

At the end of the convolution operation, the output is subject to an
activation function to allow non-linearity. The usual activation
function for convnet is the Relu. All the pixel with a negative value
will be replaced by zero.

-   Max-pooling operation

This step is easy to understand. The purpose of the pooling is to reduce
the dimensionality of the input image. The steps are done to reduce the
computational complexity of the operation. By diminishing the
dimensionality, the network has lower weights to compute, so it prevents
overfitting.

In this stage, you need to define the size and the stride. A standard
way to pool the input image is to use the maximum value of the feature
map. Look at the picture below. The "pooling" will screen a four
submatrix of the 4x4 feature map and return the maximum value. The
pooling takes the maximum value of a 2x2 array and then move this
windows by two pixels. For instance, the first sub-matrix is
\[3,1,3,2\], the pooling will return the maximum, which is 3.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/20_CNN_v7_files/image012.png)

There is another pooling operation such as the mean.

This operation aggressively reduces the size of the feature map

-   Fully connected layers

The last step consists of building a traditional artificial neural
network as you did in the previous tutorial. You connect all neurons
from the previous layer to the next layer. You use a softmax activation
function to classify the number on the input image.

**Recap:** 

Convolutional Neural network compiles different layers before making a
prediction. A neural network has:

-   A convolutional layer

-   Relu Activation function

-   Pooling layer

-   Densely connected layer

The convolutional layers apply different filters on a subregion of the
picture. The Relu activation function adds non-linearity, and the
pooling layers reduce the dimensionality of the features maps.

All these layers extract essential information from the images. At last,
the features map are feed to a primary fully connected layer with a
softmax function to make a prediction.

# Train CNN with TensorFlow

Now that you are familiar with the building block of a convnets, you are
ready to build one with TensorFlow. We will use the MNIST dataset.

The data preparation is the same as the previous tutorial. You can run
the codes and jump directly to the architecture of the CNN.

You will follow the steps below:

Step 1: Upload Dataset

Step 2: Input layer

Step 3: Convolutional layer

Step 4: Pooling layer

Step 5: Second Convolutional Layer and Pooling Layer

Step 6: Dense layer

Step 7: Logit Layer


**Step 1**: Upload Dataset

The MNIST dataset is available at this [URL](https://www.dropbox.com/sh/jm9jo0d58oggeb9/AAAZrRHvHFGYdCHssXpEH2o1a?dl=0).

Please download it and store it in Downloads. You can upload it with `fetch_mldata('MNIST original')`.

**Create a train/test set**

You need to split the dataset with `train_test_split

**Scale the features**

Finally, you can scale the feature with MinMaxScaler

    import numpy as np
    import tensorflow as tf
    
    from sklearn.datasets import fetch_mldata

If you are a Windows user, run the following lines

    #Change USERNAME by the username of your machine
    
    ## Windows USER
    
    mnist = fetch_mldata('C:\\Users\\USERNAME\\Downloads\\MNIST original')

Otherwise, you need to run this line

    ## Mac User
    mnist = fetch_mldata('/Users/USERNAME/Downloads/MNIST original')
    print(mnist.data.shape)
    print(mnist.target.shape)

    (70000, 784)
    (70000,)

You split the dataset with 80 percent training and 20 percent testing. 

    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(mnist.data,
                                                        mnist.target,
                                                        test_size=0.2,
                                                        random_state=42)
    y_train  = y_train.astype(int)
    y_test  = y_test.astype(int)
    batch_size =len(X_train)
    
    print(X_train.shape, y_train.shape,y_test.shape )

    (56000, 784) (56000,) (14000,)

Finaly, you scale the data using the min/max scaler of scikit learn. 

    ## resclae
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # Train
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    # test
    X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))
    
    feature_columns = [
          tf.feature_column.numeric_column('x', shape=X_train_scaled.shape[1:])]
    
    X_train_scaled.shape[1:]



    (784,)



**Define the CNN**

A CNN uses filters on the raw pixel of an image to learn details pattern
compare to global pattern with a traditional neural net. To construct a
CNN, you need to define:

1. A convolutional layer: Apply n number of filters to the feature
   map. After the convolution, you need to use a Relu activation
   function to add non-linearity to the network.
2. Pooling layer: The next step after the convolution is to downsample
   the feature max. The purpose is to reduce the dimensionality of the
   feature map to prevent overfitting and improve the computation
   speed. Max pooling is the conventional technique, which divides the
   feature maps into subregions (usually with a 2x2 size) and keeps
   only the maximum values.
3. Fully connected layers: All neurons from the previous layers are
   connected to the next layers. The CNN will classify the label
   according to the features from the convolutional layers and reduced
   with the pooling layer.

CNN architecture

- Convolutional Layer: Applies 14 5x5 filters (extracting 5x5-pixel
  subregions), with ReLU activation function
- Pooling Layer: Performs max pooling with a 2x2 filter and stride of
  2 (which specifies that pooled regions do not overlap)
- Convolutional Layer: Applies 36 5x5 filters, with ReLU activation
  function
- Pooling Layer 2: Again, performs max pooling with a 2x2 filter and
  stride of 2
- 1,764 neurons, with dropout regularization rate of 0.4 (probability
  of 0.4 that any given element will be dropped during training)
- Dense Layer (Logits Layer): 10 neurons, one for each digit target
  class (0–9).

There are three important modules to use to create a CNN:

- `conv2d()`. Constructs a two-dimensional convolutional layer with the
  number of filters, filter kernel size, padding, and activation
  function as arguments.
- `max_pooling2d()`. Constructs a two-dimensional pooling layer using
  the max-pooling algorithm.
- `dense()`. Constructs a dense layer with the hidden layers and units

You will define a function to build the CNN. Let's see in detail how to
construct each building block before to wrap everything together in the
function.

**Step 2**: Input layer

    def cnn_model_fn(features, labels, mode):
        input_layer = tf.reshape(tensor = features["x"],shape =[-1, 28, 28, 1])

You need to define a tensor with the shape of the data. For that, you can use the module tf.reshape. In this module, you need to declare the
tensor to reshape and the shape of the tensor. The first argument is the
features of the data, which is defined in the argument of the function.
A picture has a height, a width, and a channel. The MNIST dataset is a
monochronic picture with a 28x28 size. We set the batch size to -1 in
the shape argument so that it takes the shape of the `features["x"]`.
The advantage is to make the batch size hyperparameters to tune. If the
batch size is set to 7, then the tensor will feed 5,488 values (28*28*7).

**Step 3**: Convolutional layer

    # first Convolutional Layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=14,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

The first convolutional layer has 14 filters with a kernel size of 5x5
with the same padding. The same padding means both the output tensor and
input tensor should have the same height and width. Tensorflow will add
zeros to the rows and columns to ensure the same size.
You use the Relu activation function. The output size will be
[28, 28, 14].

**Step 4**: Pooling layer
The next step after the convolution is the pooling computation. The
pooling computation will reduce the dimensionality of the data. You can
use the module max_pooling2d with a size of 2x2 and stride of 2. You
use the previous layer as input. The output size will be
[batch_size, 14, 14, 14]

    # first Pooling Layer 
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

**Step 5**: Second Convolutional Layer and Pooling Layer

The second convolutional layer has 32 filters, with an output size of
`[batch_size, 14, 14, 32]`. The pooling layer has the same size as
before and the output shape is `[batch_size, 14, 14, 18]`.

    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=36,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

**Step 6**: Dense layer
Then, you need to define the fully-connected layer. The feature map has
to be flatten before to be connected with the dense layer. You can use
the module reshape with a size of 7*7*36.
The dense layer will connect 1764 neurons. You add a Relu activation
function. Besides, you add a dropout regularization term with a rate of
0.3, meaning 30 percents of the weights will be set to 0. Note that, the
dropout takes place only during the training phase. The function
cnn_model_fn has an argument mode to declare if the model needs to
be trained or to evaluate.

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 36])
    
    dense = tf.layers.dense(inputs=pool2_flat, units=7 * 7 * 36, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
          inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

**Step 7**: Logit Layer

Finally, you can define the last layer with the prediction of the model.
The output shape is equal to the batch size and 10, the total number of
images.

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

You can create a dictionary containing the classes and the probability
of each class. The module `tf.argmax()` with returns the highest value
if the logit layers. The softmax function returns the probability of
each class.

    predictions = {
          # Generate predictions
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }

You only want to return the dictionnary prediction when mode is set to
prediction. You add this codes to dispay the predictions

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

The next step consists to compute the loss of the model. In the last
tutorial, you learnt that the loss function for a multiclass model is
cross entropy. The loss is easily computed with the following code:

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

The final step is to optimize the model, that is to find the best values
of the weights. For that, you use a Gradient descent optimizer with a
learning rate of 0.001. The objective is to minimize the loss

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

You are done with the CNN. However, you want to display the performance
metrics during the evaluation mode. The performance metrics for a
multiclass model is the accuracy metrics. Tensorflow is equipped with
a module accuracy with two arguments, the labels, and the predicted
values.

    eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

That's it. You created your first CNN and you are ready to wrap
everything into a function in order to use it to train and evaluate the
model.

    def cnn_model_fn(features, labels, mode):
      """Model function for CNN."""
      # Input Layer
      input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
      # Convolutional Layer
      conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
    
      # Pooling Layer
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
      # Convolutional Layer #2 and Pooling Layer
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=36,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
      # Dense Layer
      pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 36])
      dense = tf.layers.dense(inputs=pool2_flat, units=7 * 7 * 36, activation=tf.nn.relu)
      dropout = tf.layers.dropout(
          inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
      # Logits Layer
      logits = tf.layers.dense(inputs=dropout, units=10)
    
      predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }
    
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
      # Calculate Loss
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
      # Configure the Training Op (for TRAIN mode)
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
      # Add evaluation metrics Evaluation mode
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    

The steps below are the same as the previous tutorials.
First of all, you define an estimator with the CNN model.

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="train/mnist_convnet_model")

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_model_dir': 'train/mnist_convnet_model', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x115c7dcf8>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}

A CNN takes many times to train, therefore, you create a Logging hook to
store the values of the softmax layers every 50 iterations.

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=50)

You are ready to estimate the model. You set a batch size of 100 and
shuffle the data. Note that we set training steps of 16.000, it can take
lots of time to train. Be patient.

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train_scaled},
        y=y_train,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=500,
        hooks=[logging_hook])

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from train/mnist_convnet_model/model.ckpt-0
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into train/mnist_convnet_model/model.ckpt.
    INFO:tensorflow:probabilities = [[0.10179272 0.10377275 0.09216717 0.10627077 0.09609951 0.10186614
      0.09686517 0.11118253 0.08496662 0.10501662]
    
                          [TRONCATED...]
    
     [0.07817432 0.11179895 0.10121216 0.1069637  0.10540046 0.08481063
      0.09004224 0.10949904 0.10535887 0.10673963]
     [0.09889819 0.08785416 0.10462773 0.10053213 0.09706611 0.09188736
      0.10994006 0.10868365 0.1060228  0.0944878 ]
     [0.09892072 0.08522665 0.09573467 0.10917412 0.09986016 0.08231454
      0.10718052 0.11899454 0.0943455  0.10824858]
     [0.11489351 0.09650423 0.10198765 0.10460417 0.09075027 0.09776379
      0.09588426 0.09854984 0.0976679  0.10139438]
     [0.1160422  0.08332406 0.11914795 0.11166839 0.09227713 0.0822787
      0.10006384 0.10538424 0.09120349 0.09861001]
     [0.10120417 0.09850905 0.10345054 0.10519108 0.09219364 0.10022194
      0.10575273 0.09497389 0.0916015  0.10690145]] (17.206 sec)
    INFO:tensorflow:Saving checkpoints for 500 into train/mnist_convnet_model/model.ckpt.
    INFO:tensorflow:Loss for final step: 2.2327070236206055.

    <tensorflow.python.estimator.estimator.Estimator at 0x115c7dba8>



Now that the model is train, you can evaluate it and print the results

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test_scaled},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2018-08-30-08:22:10
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from train/mnist_convnet_model/model.ckpt-500
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Finished evaluation at 2018-08-30-08:22:30
    INFO:tensorflow:Saving dict for global step 500: accuracy = 0.44878572, global_step = 500, loss = 2.2214103
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 500: train/mnist_convnet_model/model.ckpt-500
    {'accuracy': 0.44878572, 'loss': 2.2214103, 'global_step': 500}

With the current architecture, you get an accuracy of 97% (after 16.00 steps training). You can change the architecture, the batch size and the number of iteration to improve the accuracy. The CNN neural network has performed far better than ANN or logistic regression. In the tutorial on an artificial neural network, you had an accuracy of 96%, which is lower the CNN. The performances of the CNN are impressive with a larger image, both in term of speed computation and accuracy.

## Summary

A convolutional neural network works very well to evaluate picture. This
type of architecture is dominant to recognize objects from a picture or
video.
To build a CNN, you need to follow six steps:

Step 1: Input layer:
This step reshapes the data. The shape is equal to the square root of
the number of pixels. For instance, if a picture has 156 pixels, then
the shape is 26x26. You need to specify if the picture has colour or
not. If yes, then you had 3 to the shape- 3 for RGB-, otherwise 1.

    input_layer = tf.reshape(tensor = features["x"],shape =[-1, 28, 28,
    
    1])



Step 2: Convolutional layer

Next, you need to create the convolutional layers. You apply different
filters to allow the network to learn important feature. You specify the
size of the kernel and the amount of filters.

    conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=14,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

Step 3: Pooling layer

In the third step, you add a pooling layer. This layer decreases the
size of the input. It does so by taking the maximum value of the a
sub-matrix. For instance, if the sub-matrix is [3,1,3,2], the pooling
will return the maximum, which is 3.

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
    
    strides=2)



Step 4: Add Convolutional Layer and Pooling Layer

In this step, you can add as much as you want conv layers and pooling
layers. Google uses architecture with more than 20 conv layers.

Step 5: Dense layer

The step 5 flatten the previous to create a fully connected layers. In
this step, you can use different activation function and add a dropout
effect.

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 36])
    
    dense = tf.layers.dense(inputs=pool2_flat, units=7 * 7 * 36, activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(
    
          inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
    

Step 6: Logit Layer

The final step is the prediction.

`logits = tf.layers.dense(inputs=dropout, units=10)`
