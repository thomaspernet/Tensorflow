
# Tensorboard Tutorial: Graph Visualization with Example 

## What is TensorBoard


Tensorboard is the interface used to visualize the graph and other tools
to understand, debug, and optimize the model.

**Example**

The image below comes from the graph you will generate in this tutorial.
It is the main panel:



![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/10_tensorbaord_files/image54_1.png)



From the picture below, you can see the panel of Tensorboard. The panel
contains different tabs, which are linked to the level of information
you add when you run the model.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/10_tensorbaord_files/image54_2.png)

-   Scalars: Show different useful information during the model training

-   Graphs: Show the model

-   Histogram: Display weights with a histogram

-   Distribution: Display the distribution of the weight

-   Projector: Show Principal component analysis and T-SNE algorithm.
    The technique uses for dimensionality reduction

During this tutorial, you will train a simple deep learning model. You
will learn how it works in a future tutorial.

If you look at the graph, you can understand how the model work.

1.  Enqueue the data to the model: Push an amount of data equal to the
    batch size to the model, i.e., Number of data feed after each
    iteration

2.  Feed the data to the Tensors

3.  Train the model

4.  Display the number of batches during the training. Save the model on
    the disk.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/10_tensorbaord_files/image54_3.png)

The basic idea behind tensorboard is that neural network can be
something known as a black box and we need a tool to inspect what’s
inside this box. You can imagine tensorboard as a flashlight to start
dive into the neural network.

It helps to understand the dependencies between operations, how the
weights are computed, displays the loss function and much other useful
information. When you bring all these pieces of information together,
you have a great tool to debug and find how to improve the model.

To give you an idea of how useful the graph can be, look at the picture
below:

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/10_tensorbaord_files/image54_4.png)

A neural network decides how to connect the different "neurons" and how
many layers before the model can predict an outcome. Once you have
defined the architecture, you not only need to train the model but also
a metrics to compute the accuracy of the prediction. This metric is
referred to as a **loss function.** The objective is to minimize the
loss function. In different words, it means the model is making fewer
errors. All machine learning algorithms will repeat many times the
computations until the loss reach a flatter line. To minimize this loss
function, you need to define a **learning rate.** It is the speed you
want the model to learn. If you set a learning rate too high, the model
does not have time to learn anything. This is the case in the left
picture. The line is moving up and down, meaning the model predicts with
pure guess the outcome. The picture on the right shows that the loss is
decreasing over iteration until the curve got flatten, meaning the model
found a solution.

TensorBoard is a great tool to visualize such metrics and highlight
potential issues. The neural network can take hours to weeks before they
find a solution. TensorBoard updates the metrics very often. In this
case, you don’t need to wait until the end to see if the model trains
correctly. You can open TensorBoard check how the training is going and
make the appropriate change if necessary.

In this tutorial, you will learn how to open TensorBoard from the
terminal for MacOS and the Command line for Windows.

The code will be explained in a future tutorial, the focus here is on
TensorBoard.

First, you need to import the libraries you will use during the training

    ## Import the library
    import tensorflow as tf
    import numpy as np

You create the data. It is an array of 10000 rows and 5 columns

    X_train = (np.random.sample((10000,5)))
    y_train =  (np.random.sample((10000,1)))
    X_train.shape

**Output**

    (10000, 5)

The codes below transform the data and create the model.

Note that the learning rate is equal to `0.1`. If you change this rate
to a higher value, the model will not find a solution. This is what
happened on the left side of the above picture.

During most of the TensorFlow tutorials, you will use TensorFlow
estimator. This is TensorFlow API that contains all the mathematical
computations.

To create the log files, you need to specify the path. This is done with
the argument `model_dir`.

In the example below, you store the model inside the working directory,
i.e., where you store the notebook or python file. Inside this path,
TensorFlow will create a folder called `train` with a child folder name
`linreg`.

    feature_columns = [
    
          tf.feature_column.numeric_column('x', shape=X_train.shape[1:])]
    
    DNN_reg = tf.estimator.DNNRegressor(
        feature_columns=feature_columns,
    
    # Indicate where to store the log file
        model_dir='train/linreg',
        hidden_units=[500, 300],
        optimizer=tf.train.ProximalAdagradOptimizer(
          learning_rate=0.1,
          l1_regularization_strength=0.001
        )
    
    )

**Output**

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_model_dir': 'train/linreg', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x1818e63828>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}

The last step consists to train the model. During the training,
TensorFlow writes information in the model directory.

    # Train the estimator
    train_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=y_train, shuffle=False,num_epochs=None)
    
    DNN_reg.train(train_input,steps=3000) 



**For MacOS user**

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/10_tensorbaord_files/image055.png)

**For Windows user**

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/10_tensorbaord_files/image55_2.png)

You can see this information in the TensorBoard.

Now that you have the log events written, you can open Tensorboard.
Tensorboad runs on port 6006 (Jupyter runs on port 8888). You can use
the `Terminal` for MacOs user or `Anaconda prompt` for Windows user.

**For MacOS user**

    # Different for you
    cd /Users/Thomas/Dropbox/Learning/tuto_TF
    source activate hello-tf!

The notebook is stored in the path `/Users/Thomas/Dropbox/Learning/tuto_TF`

**For Windows users**

    cd C:\Users\Admin\Anaconda3
    activate hello-tf

The notebook is stored in the path `C:\Users\Admin\Anaconda3`

To launch Tensorboard, you can use this code

**For MacOS user**

`tensorboard --logdir=./train/linreg`

**For Windows users**\
`tensorboard --logdir=.\\train\\linreg`

Tensorboard is located in this URL: http://localhost:6006

It could also be located at the following location.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/10_tensorbaord_files/image55_1.png)

Copy and paste the URL into your favorite browser. You should see this:

Note that, we will learn how to read the graph in the tutorial dedicated
to the deep learning.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/10_tensorbaord_files/image056.png)

If you see something like this:

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/10_tensorbaord_files/image057.png)

It means Tensorboard cannot find the log file. Make sure you point the
cd to the right path or double check if the log event has been creating.
If not, re-run the code.

If you want to close TensorBoard Press CTRL+C

Hat Tip: Check your anaconda prompt for the current working directory,

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/10_tensorbaord_files/image057_2.png)

The log file should be created at `C:\\Users\\Admin`

## Summary:


TensorBoard is a great tool to visualize your model. Besides, many
metrics are displayed during the training, such as the loss, accuracy or
weights.

To activate Tensorboard, you need to set the path of your file:

-   `cd /Users/Thomas/Dropbox/Learning/tuto_TF`

`Activate Tensorflow’s environment`

-   `activate hello-tf`

`Launch Tensorboard`

-   `tensorboard --logdir=.+ PATH`


