
# Linear Classifier TensorFlow Binary, 

# What is Linear Classifier?

The two most common supervised learning tasks are linear regression and
linear classifier. Linear regression predicts a value while the linear
classifier predicts a class. This tutorial is focused on Linear
Classifier.

Classification problems represent roughly 80 percent of the machine
learning task. Classification aims at predicting the probability of each
class given a set of inputs. The label (i.e., the dependent variable) is
a discrete value, called a class.

1.  If the label has only two classes, the learning algorithm is a
    binary classifier.

2.  Multiclass classifier tackles labels with more than two classes.

For instance, a typical binary classification problem is to predict the
likelihood a customer makes a second purchase. Predict the type of
animal displayed on a picture is multiclass classification problem since
there are more than two varieties of animal existing.

The theoretical part of this tutorial puts primary focus on the binary
class. You will learn more about the multiclass output function in a
future tutorial.

In this tutorial, you will learn

## How Binary classifier works?

You learned in the previous tutorial that a function is composed of two
kind of variables, a dependent variable and a set of features
(independent variables). In the linear regression, a dependent variable
is a real number without range. The primary objective is to predict its
value by minimizing the mean squared error.

For a binary task, the label can have had two possible integer values.
In most case, it is either `[0,1]` or `[1,2]`. For instance, the
objective is to predict whether a customer will buy a product or not.
The label is defined as follow:

-   `Y = 1` (customer purchased the product)

-   `Y = 0` (customer does not purchase the product)

The model uses the features `X` to classify each customer in the most
likely class he belongs to, namely, potential buyer or not.

The probability of success is computed with **logistic regression**. The
algorithm will compute a probability based on the feature `X` and
predicts a success when this probability is above 50 percent. More
formally, the probability is calculated as follow:

$$P(Y = 1|x) = \frac{1}{1 + exp( - (\theta^{T}x + b))}$$

where $\theta$ is the set of weights, $x$ the features and b the bias.

The function can be decomposed into two parts:

-   The linear model

-   The logistic function

**Linear model**

You are already familiar with the way the weights are computed. Weights
are computed using a dot product: $\theta^{T}x + b$ that is $\sum_{i=0}^{n} x_i w_i +b$. `Y` is a linear function of all the
features $x_i$. If the model does not have features, the prediction is
equal to the bias, $b$.

The weights indicate the direction of the correlation between the
features $x_i$ and the label $Y$. A positive correlation increases the
probability of the positive class while a negative correlation leads the
probability closer to 0, (i.e., negative class).

The linear model returns only real number, which is inconsistent with
the probability measure of range `[0,1]`. The logistic function is
required to convert the linear model output to a probability,

**Logistic function**

The logistic function, or sigmoid function, has an S-shape and the
output of this function is always between 0 and 1.

$$\frac{1}{1 + exp( - t)}$$

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/17_logit_v4_files/image011.png)

It is easy to substitute the output of the linear regression into the
sigmoid function. It results in a new number with a probability between
0 and 1.

The classifier can transform the probability into a class

-   Values between 0 to 0.49 become class 0

-   Values between 0.5 to 1 become class 1

## How to Measure the performance of Linear Classifier?

### Accuracy

The overall performance of a classifier is measured with the accuracy
metric. Accuracy collects all the correct values divided by the total
number of observations. For instance, an accuracy value of 80 percent
means the model is correct in 80 percent of the cases.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/17_logit_v4_files/image013.png)

You can note a shortcoming with this metric, especially for imbalance
class. An imbalance dataset occurs when the number of observations per
group is not equal. Let’s say; you try to classify a rare event with a
logistic function. Imagine the classifier tries to estimate the death of
a patient following a disease. In the data, 5 percent of the patients
pass away. You can train a classifier to predict the number of death and
use the accuracy metric to evaluate the performances. If the classifier
predicts 0 death for the entire dataset, it will be correct in 95
percent of the case.

### Confusion matrix

A better way to assess the performance of a classifier is to look at the
confusion matrix.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/17_logit_v4_files/image015.png)

The confusion matrix visualizes the accuracy of a classifier by
comparing the actual and predicted classes. The binary confusion matrix
is composed of squares:

-   TP: True Positive: Predicted values correctly predicted as actual
    positive

-   FP: Predicted values incorrectly predicted an actual positive. i.e.,
    Negative values predicted as positive

-   FN: False Negative: Positive values predicted as negative

-   TN: True Negative: Predicted values correctly predicted as actual
    negative

From the confusion matrix, it is easy to compare the actual class and
predicted class.

### Precision and Sensitivity

The confusion matrix provides a good insight into the true positive and
false positive. In some case, it is preferable to have a more concise
metric.

**Precision**

The precision metric shows the accuracy of the positive class. It
measures how likely the prediction of the positive class is correct.

$$Precision = \frac{\text{TP}}{TP + FP}$$

The maximum score is 1 when the classifier perfectly classifies all the
positive values. Precision alone is not very helpful because it ignores
the negative class. The metric is usually paired with Recall metric.
Recall is also called sensitivity or true positive rate.

**Sensitivity**

Sensitivity computes the ratio of positive classes correctly detected.
This metric gives how good the model is to recognize a positive class.

$$Recall = \frac{\text{TP}}{TP + FN}$$

## Linear Classifier with TensorFlow

For this tutorial, we will use the census dataset. The purpose is to use
the variables in the census dataset to predict the income level. Note
that the income is a binary variable

-   with a value of 1 if the income &gt; 50k

-   0 if income &lt; 50k.

This variable is your label

This dataset includes eight categorical variables:

-   workplace

-   education

-   marital

-   occupation

-   relationship

-   race

-   sex

-   native\_country

moreover, six continuous variables:

-   age

-   fnlwgt

-   education\_num

-   capital\_gain

-   capital\_loss

-   hours\_week

Through this example, you will understand how to train a linear
classifier with TensorFlow estimator and how to improve the accuracy
metric.

We will proceed as follow:

Step 1: Import the data

Step 2: Data Conversion

Step 3: Train the classifier

Step 4: Improve the model

Step 5: Hyperparameter:Lasso & Ridge

**Step 1**: Import the data

You first import the libraries used during the tutorial.

import tensorflow as tf\
import pandas as pd

Next, you import the data from the archive of UCI and defines the
columns names. You will use the `COLUMNS` to name the columns in a
pandas data frame.

Note that you will train the classifier using a Pandas dataframe.


```python
import pandas as pd
import tensorflow as tf
## Define path data
COLUMNS = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital',
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_week', 'native_country', 'label']
PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
PATH_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
```

    /Users/Thomas/anaconda3/envs/hello-tf/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
      return f(*args, **kwds)


The data stored online are already divided between a train set and test
set.


```python
df_train = pd.read_csv(PATH, skipinitialspace=True, names = COLUMNS, index_col=False)
df_test = pd.read_csv(PATH_test,skiprows = 1, skipinitialspace=True, names = COLUMNS, index_col=False)
```

The train set contains 32,561 observations and the test set 16,281


```python
print(df_train.shape, df_test.shape)
print(df_train.dtypes)

```

    (32561, 15) (16281, 15)
    age                int64
    workclass         object
    fnlwgt             int64
    education         object
    education_num      int64
    marital           object
    occupation        object
    relationship      object
    race              object
    sex               object
    capital_gain       int64
    capital_loss       int64
    hours_week         int64
    native_country    object
    label             object
    dtype: object


Tensorflow requires a Boolean value to train the classifier. You need to
cast the values from string to integer. The label is store as an object,
however, you need to convert it into a numeric value. The code below
creates a dictionary with the values to convert and loop over the column
item. Note that you perform this operation twice, one for the train
test, one for the test set


```python
label = {'<=50K': 0,'>50K': 1}
df_train.label = [label[item] for item in df_train.label]
label_t = {'<=50K.': 0,'>50K.': 1}
df_test.label = [label_t[item] for item in df_test.label]
```

In the train data, there are 24,720 incomes lower than 50k and 7841 above. The ratio is almost the same for the test set. Please refer this tutorial on Facets for more.


```python
print(df_train["label"].value_counts())
### The model will be correct in atleast 70% of the case
print(df_test["label"].value_counts())
## Unbalanced label
print(df_train.dtypes)
```

    0    24720
    1     7841
    Name: label, dtype: int64
    0    12435
    1     3846
    Name: label, dtype: int64
    age                int64
    workclass         object
    fnlwgt             int64
    education         object
    education_num      int64
    marital           object
    occupation        object
    relationship      object
    race              object
    sex               object
    capital_gain       int64
    capital_loss       int64
    hours_week         int64
    native_country    object
    label              int64
    dtype: object


**Step 2**: Data Conversion 

A few steps are required before you train a linear classifier with
Tensorflow. You need to prepare the features to include in the model. In
the benchmark regression, you will use the original data without
applying any transformation.

The estimator needs to have a list of
features to train the model. Hence, the column's data requires to be
converted into a tensor.

A good practice is to define two lists of features based on their type
and then pass them in the `feature_columns` of the estimator.

You will begin by converting continuous features, then define a bucket
with the categorical data.

The features of the dataset have two formats:

-   Integer

-   Object

Each feature is listed in the next two variables as per their types.


```python
## Add features to the bucket: 
### Define continuous list
CONTI_FEATURES  = ['age', 'fnlwgt','capital_gain', 'education_num', 'capital_loss', 'hours_week']
### Define the categorical list
CATE_FEATURES = ['workclass', 'education', 'marital', 'occupation', 'relationship', 'race', 'sex', 'native_country']
```

The `feature_column` is equipped with an object `numeric_column` to help
in the transformation of the continuous variables into tensor. In the
code below, you convert all the variables from CONTI\_FEATURES into a
tensor with a numeric value. This is compulsory to construct the model.
All the independent variables need to be converted into the proper type
of tensor.

Below we write a code to let you see what is happening behind
`feature_column.numeric_column`. We will print the converted value for
age It is for explanatory purpose, hence there is no need to understand
the python code. You can refer to the official documentation to
understand the codes.



```python
def print_transformation(feature = "age", continuous = True, size = 2): 
    #X = fc.numeric_column(feature)
    ## Create feature name
    feature_names = [
    feature]

    ## Create dict with the data
    d = dict(zip(feature_names, [df_train[feature]]))

    ## Convert age
    if continuous == True:
        c = tf.feature_column.numeric_column(feature)
        feature_columns = [c]
    else: 
        c = tf.feature_column.categorical_column_with_hash_bucket(feature, hash_bucket_size=size) 
        c_indicator = tf.feature_column.indicator_column(c)
        feature_columns = [c_indicator]
    
## Use input_layer to print the value
    input_layer = tf.feature_column.input_layer(
        features=d,
        feature_columns=feature_columns
        )
    ## Create lookup table
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(input_layer, zero)
    ## Return lookup tble
    indices = tf.where(where)
    values = tf.gather_nd(input_layer, indices)
    ## Initiate graph
    sess = tf.Session()
    ## Print value
    print(sess.run(input_layer))
```


```python
print_transformation(feature = "age", continuous = True)
```

    [[39.]
     [50.]
     [38.]
     ...
     [58.]
     [22.]
     [52.]]


The values are exactly the same as in df_train


```python
continuous_features = [tf.feature_column.numeric_column(k) for k in CONTI_FEATURES]
```

According to TensorFlow documentation, there are different ways to
convert categorical data. If the vocabulary list of a feature is known
and does not have plenty of values, it is possible to create the
categorical column with `categorical_column_with_vocabulary_list`. It
will assign to all unique vocabulary list an ID.

For instance, if a variable `statu`s has three distinct values:

-   Husband

-   Wife

-   Single

Then three ID will be attributed. For instance, Husband will have the ID
1, Wife the ID 2 and so on.

For illustration purpose, you can use this code to convert an object
variable to a categorical column in TensorFlow.

The feature sex can only have two value: male or female. When we will
convert the feature sex, Tensorflow will create 2 new columns, one for
male and one for female. If the sex is equal to male, then the new
column male will be equal to 1 and female to 0. This example is
displayed in the table below:

| rows | sex    | after transformation | male | female |
|------|--------|----------------------|------|--------|
| 1    | male   | =>                   | 1    | 0      |
| 2    | male   | =>                   | 1    | 0      |
| 3    | female | =>                   | 0    | 1      |

In tensorflow:

print\_transformation(feature = "sex", continuous = False, size = 2)


```python
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'])
```

Below, we added Python code to print the encoding. Again, you don’t need to understand the code, the purpose is to see the transformation

However, a faster way to transform the data is to use the method
`categorical_column_with_hash_bucket`. Altering string variables in a sparse matrix will be useful. A sparse matrix is a matrix with mostly
zero. The method takes care of everything. You only need to specify the
number of buckets and the key column. The number of buckets is the
maximum amount of groups that Tensorflow can create. The key column is
simply the name of the column to convert.

In the code below, you create a loop over all the categorical features.


```python
categorical_features = [tf.feature_column.categorical_column_with_hash_bucket(k, hash_bucket_size=1000) for k in CATE_FEATURES]
```

**Step 3**: Train the Classifier

TensorFlow currently provides an estimator for the linear regression and
linear classification.

-   Linear regression: `LinearRegressor`

-   Linear classification: `LinearClassifier`

The syntax of the linear classifier is the same as in the tutorial on
linear regression except for one argument, n\_class. You need to define
the feature column, the model directory and, compare with the linear
regressor; you have the define the number of class. For a logit
regression, it the number of class is equal to 2.

The model will compute the weights of the columns contained in
`continuous_features` and `categorical_features`.


```python
model = tf.estimator.LinearClassifier(
    n_classes = 2,
    model_dir="ongoing/train", 
    feature_columns=categorical_features+ continuous_features)
```

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_model_dir': 'ongoing/train', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0xb203082b0>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}


Now that the classifier is defined, you can create the input function.
The method is the same as in the linear regressor tutorial. Here, you
use a batch size of 128 and you shuffle the data.


```python
FEATURES = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_week', 'native_country']
LABEL= 'label'
def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
       x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
       y = pd.Series(data_set[LABEL].values),
       batch_size=n_batch,   
       num_epochs=num_epochs,
       shuffle=shuffle)


```

You create a function with the arguments required by the linear estimator, i.e., number of epochs, number of batches and shuffle the dataset or note. Since you use the Pandas method to pass the data into the model, you need to define the X variables as a pandas data frame. Note that you loop over all the data stored in FEATURES. 

Let’s train the model with the object model.train. You use the function
previously defined to feed the model with the appropriate values. Note
that you set the batch size to 128 and the number of epochs to None. The
model will be trained over a thousand steps.


```python
model.train(input_fn=get_input_fn(df_train, 
                                      num_epochs=None,
                                      n_batch = 128,
                                      shuffle=False),
                                      steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into ongoing/train/model.ckpt.
    INFO:tensorflow:loss = 88.722855, step = 1
    INFO:tensorflow:global_step/sec: 95.9134
    INFO:tensorflow:loss = 52583.64, step = 101 (1.044 sec)
    INFO:tensorflow:global_step/sec: 167.726
    INFO:tensorflow:loss = 25203.816, step = 201 (0.596 sec)
    INFO:tensorflow:global_step/sec: 162.827
    INFO:tensorflow:loss = 54924.312, step = 301 (0.614 sec)
    INFO:tensorflow:global_step/sec: 226.156
    INFO:tensorflow:loss = 68509.31, step = 401 (0.443 sec)
    INFO:tensorflow:global_step/sec: 143.237
    INFO:tensorflow:loss = 9151.754, step = 501 (0.701 sec)
    INFO:tensorflow:global_step/sec: 140.458
    INFO:tensorflow:loss = 34576.06, step = 601 (0.710 sec)
    INFO:tensorflow:global_step/sec: 131.307
    INFO:tensorflow:loss = 36047.117, step = 701 (0.764 sec)
    INFO:tensorflow:global_step/sec: 150.417
    INFO:tensorflow:loss = 22608.148, step = 801 (0.665 sec)
    INFO:tensorflow:global_step/sec: 162.276
    INFO:tensorflow:loss = 22201.918, step = 901 (0.615 sec)
    INFO:tensorflow:Saving checkpoints for 1000 into ongoing/train/model.ckpt.
    INFO:tensorflow:Loss for final step: 5444.363.





    <tensorflow.python.estimator.canned.linear.LinearClassifier at 0xb202e4668>



Note that the loss decreased subsequently during the last 100 steps, i.e., from 901 to 1000.

The final loss after one thousand iterations is 5444. You can estimate
your model on the test set and see the performance. To evaluate the
performance of your model, you need to use the object evaluate. You feed
the model with the test set and set the number of epochs to 1, i.e., the
data will go to the model only one time.


```python
model.evaluate(input_fn=get_input_fn(df_test, 
                                      num_epochs=1,
                                      n_batch = 128,
                                      shuffle=False),
                                      steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2018-08-29-19:10:30
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ongoing/train/model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Evaluation [100/1000]
    INFO:tensorflow:Finished evaluation at 2018-08-29-19:10:33
    INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.7615626, accuracy_baseline = 0.76377374, auc = 0.63300294, auc_precision_recall = 0.50891197, average_loss = 47.12155, global_step = 1000, label/mean = 0.23622628, loss = 5993.6406, precision = 0.49401596, prediction/mean = 0.18454961, recall = 0.38637546
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 1000: ongoing/train/model.ckpt-1000





    {'accuracy': 0.7615626,
     'accuracy_baseline': 0.76377374,
     'auc': 0.63300294,
     'auc_precision_recall': 0.50891197,
     'average_loss': 47.12155,
     'label/mean': 0.23622628,
     'loss': 5993.6406,
     'precision': 0.49401596,
     'prediction/mean': 0.18454961,
     'recall': 0.38637546,
     'global_step': 1000}



TensorFlow returns all the metrics you learnt in the theoretical part.
Without surprise, the accuracy is large due to the unbalanced label.
Actually, the model performs slightly better than a random guess.
Imagine the model predict all household with income lower than 50K, then
the model has an accuracy of 70 percent. On a closer analysis, you can
see the prediction and recall are quite low.

**Step 4**: Improve the model

Now that you have a benchmark model, you can try to improve it, that is,
increase the accuracy. In the previous tutorial, you learned how to
improve the prediction power with an interaction term. In this tutorial,
you will revisit this idea by adding a polynomial term to the
regression.

Polynomial regression is instrumental when there is non-linearity in the
data. There are two ways to capture non-linearity in the data.

-   Add polynomial term

-   Bucketize the continuous variable into a categorical variable

**Polynomial term**

From the picture below, you can see what a polynomial regression is. It
is an equation with X variables with different power. A second-degree
polynomial regression has two variables, X and X squared. Third degree
has three variables, X, $X^{2},\ and\ X^{3}$

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/17_logit_v4_files/image020.png)

Below, we constructed a graph with two variables, `X` and `Y`. It is
obvious the relationship is not linear. If we add a linear regression,
we can see the model is unable to capture the pattern (left picture).

Now, look at the left picture from the picture below, we added five-term
to the regression (that is $y = x + x^{2} + x^{3} + x^{4} + x^{5}$. The
model now captures way better the pattern. This is the power of
polynomial regression.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/17_logit_v4_files/image021.png)

Let's go back to our example. Age is *not* in a linear relationship with
income. Early age might have a flat income close to zero because
children or young people do not work. Then it increases in working age
and decreases during retirement. It is typically an Inversed-U shape.
One way to capture this pattern is by adding a power two to the
regression.

Let’s see if it increases the accuracy.

You need to add this new feature to the dataset and in the list of
continuous feature.

You add the new variable in the train and test dataset, so it is more
convenient to write a function.


```python
def square_var(df_t, df_te, var_name = 'age'):
    df_t['new'] = df_t[var_name].pow(2) 
    df_te['new'] = df_te[var_name].pow(2) 
    return df_t, df_te
```

The function has 3 arguments:

df\_t: define the training set

df\_te: define the test set

var\_name = 'age': Define the variable to transform

You can use the object pow(2) to square the variable age. Note that the
new variable is named 'new'

Now that the function `square_var` is written, you can create the new
datasets.


```python
df_train_new, df_test_new = square_var(df_train, df_test, var_name = 'age')
```

As you can see, the new dataset has one more feature.


```python
print(df_train_new.shape, df_test_new.shape)

```

    (32561, 16) (16281, 16)


The square variable is called `new` in the dataset. You need to add it
to the list of continuous features.


```python
CONTI_FEATURES_NEW  = ['age', 'fnlwgt','capital_gain', 'education_num', 'capital_loss', 'hours_week', 'new']
continuous_features_new = [tf.feature_column.numeric_column(k) for k in CONTI_FEATURES_NEW]
```

**Note** that you changed the directory of the Graph. You can’t train
different models in the same directory. It means, you need to change the
path of the argument model\_dir. If you don’t TensorFlow will throw an
error.


```python
model_1 = tf.estimator.LinearClassifier(
    model_dir="ongoing/train1", 
    feature_columns=categorical_features+ continuous_features_new)
```

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_model_dir': 'ongoing/train1', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0xb20308438>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}



```python
FEATURES_NEW = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_week', 'native_country', 'new']
def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
       x=pd.DataFrame({k: data_set[k].values for k in FEATURES_NEW}),
       y = pd.Series(data_set[LABEL].values),
       batch_size=n_batch,   
       num_epochs=num_epochs,
       shuffle=shuffle)
```

Now that the classifier is designed with the new dataset, you can train
and evaluate the model.


```python
model_1.train(input_fn=get_input_fn(df_train, 
                                      num_epochs=None,
                                      n_batch = 128,
                                      shuffle=False),
                                      steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into ongoing/train1/model.ckpt.
    INFO:tensorflow:loss = 88.722855, step = 1
    INFO:tensorflow:global_step/sec: 65.4699
    INFO:tensorflow:loss = 70077.66, step = 101 (1.533 sec)
    INFO:tensorflow:global_step/sec: 166.451
    INFO:tensorflow:loss = 49522.082, step = 201 (0.599 sec)
    INFO:tensorflow:global_step/sec: 172.15
    INFO:tensorflow:loss = 107120.57, step = 301 (0.577 sec)
    INFO:tensorflow:global_step/sec: 135.673
    INFO:tensorflow:loss = 12814.152, step = 401 (0.741 sec)
    INFO:tensorflow:global_step/sec: 147.318
    INFO:tensorflow:loss = 19573.898, step = 501 (0.675 sec)
    INFO:tensorflow:global_step/sec: 205.764
    INFO:tensorflow:loss = 26381.986, step = 601 (0.486 sec)
    INFO:tensorflow:global_step/sec: 188.238
    INFO:tensorflow:loss = 23417.719, step = 701 (0.531 sec)
    INFO:tensorflow:global_step/sec: 226.805
    INFO:tensorflow:loss = 23946.049, step = 801 (0.441 sec)
    INFO:tensorflow:global_step/sec: 183.742
    INFO:tensorflow:loss = 3309.5786, step = 901 (0.544 sec)
    INFO:tensorflow:Saving checkpoints for 1000 into ongoing/train1/model.ckpt.
    INFO:tensorflow:Loss for final step: 28861.898.





    <tensorflow.python.estimator.canned.linear.LinearClassifier at 0xb20308ba8>




```python
model_1.evaluate(input_fn=get_input_fn(df_test_new, 
                                      num_epochs=1,
                                      n_batch = 128,
                                      shuffle=False),
                                      steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2018-08-29-19:10:49
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ongoing/train1/model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Evaluation [100/1000]
    INFO:tensorflow:Finished evaluation at 2018-08-29-19:10:51
    INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.7944229, accuracy_baseline = 0.76377374, auc = 0.6093755, auc_precision_recall = 0.54885805, average_loss = 111.0046, global_step = 1000, label/mean = 0.23622628, loss = 14119.265, precision = 0.6682401, prediction/mean = 0.09116262, recall = 0.2576703
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 1000: ongoing/train1/model.ckpt-1000





    {'accuracy': 0.7944229,
     'accuracy_baseline': 0.76377374,
     'auc': 0.6093755,
     'auc_precision_recall': 0.54885805,
     'average_loss': 111.0046,
     'label/mean': 0.23622628,
     'loss': 14119.265,
     'precision': 0.6682401,
     'prediction/mean': 0.09116262,
     'recall': 0.2576703,
     'global_step': 1000}



The squared variable improved the accuracy from 0.76 to 0.79. Let’s see
if you can do better by combining bucketization and interaction term
together.

## Bucketization and interaction

As you saw before, a linear classifier is unable to capture the
age-income pattern correctly. That is because it learns a single weight
for each feature. To make it easier for the classifier, one thing you
can do is bucket the feature. Bucketing transforms a numeric feature
into several certain ones based on the range it falls into, and each of
these new features indicates whether a person's age falls within that
range.

With these new features, the linear model can capture the relationship
by learning different weights for each bucket.

In TensorFlow, it is done with `bucketized_column`. You need to add the
range of values in the boundaries.


```python
age = tf.feature_column.numeric_column('age')
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

You already know age is non-linear with income. Another way to improve
the model is through interaction. In the word of TensorFlow, it is
feature crossing. Feature crossing is a way to create new features that
are combinations of existing ones, which can be helpful for a linear
classifier that can’t model interactions between features.

You can break down `age` with another feature like `education`. That is
is, some groups are likely to have a high income and others low (Think
about the Ph.D. student).

.


```python
education_x_occupation = [tf.feature_column.crossed_column(
    ['education', 'occupation'], hash_bucket_size=1000)]
age_buckets_x_education_x_occupation = [tf.feature_column.crossed_column(
    [age_buckets, 'education', 'occupation'], hash_bucket_size=1000)]

```

To create a cross feature column, you use crossed_column with the variables to cross in a bracket. The hash_bucket_size indicates the maximum crossing possibilities. To create interaction between variables (at least one variable needs to be categorical), you can use tf.feature_column.crossed_column. To use this object, you need to add in square bracket the variable to interact and a second argument, the bucket size. The bucket size is the maximum number of group possible within a variable. Here you set it at 1000 as you do not know the exact number of groups

`age_buckets` needs to be squared before to add it to the feature
columns. You also add the new features to the features columns and
prepare the estimator


```python
base_columns = [
    age_buckets,
]

model_imp = tf.estimator.LinearClassifier(
    model_dir="ongoing/train3", 
    feature_columns=categorical_features+base_columns+education_x_occupation+age_buckets_x_education_x_occupation)
```

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_model_dir': 'ongoing/train3', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0xb20ef8550>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}



```python
FEATURES_imp = ['age','workclass', 'education', 'education_num', 'marital',
                'occupation', 'relationship', 'race', 'sex', 'native_country', 'new']

def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
       x=pd.DataFrame({k: data_set[k].values for k in FEATURES_imp}),
       y = pd.Series(data_set[LABEL].values),
       batch_size=n_batch,   
       num_epochs=num_epochs,
       shuffle=shuffle)
```

You are ready to estimate the new model and see if it improves the
accuracy.


```python
model_imp.train(input_fn=get_input_fn(df_train_new, 
                                      num_epochs=None,
                                      n_batch = 128,
                                      shuffle=False),
                                      steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into ongoing/train3/model.ckpt.
    INFO:tensorflow:loss = 88.722855, step = 1
    INFO:tensorflow:global_step/sec: 110.894
    INFO:tensorflow:loss = 50.334488, step = 101 (0.905 sec)
    INFO:tensorflow:global_step/sec: 204.578
    INFO:tensorflow:loss = 56.153225, step = 201 (0.489 sec)
    INFO:tensorflow:global_step/sec: 201.008
    INFO:tensorflow:loss = 45.792007, step = 301 (0.495 sec)
    INFO:tensorflow:global_step/sec: 145.813
    INFO:tensorflow:loss = 37.485672, step = 401 (0.688 sec)
    INFO:tensorflow:global_step/sec: 255.157
    INFO:tensorflow:loss = 56.48449, step = 501 (0.390 sec)
    INFO:tensorflow:global_step/sec: 196.914
    INFO:tensorflow:loss = 32.528934, step = 601 (0.507 sec)
    INFO:tensorflow:global_step/sec: 190.965
    INFO:tensorflow:loss = 37.438057, step = 701 (0.529 sec)
    INFO:tensorflow:global_step/sec: 162.964
    INFO:tensorflow:loss = 61.1075, step = 801 (0.610 sec)
    INFO:tensorflow:global_step/sec: 202.747
    INFO:tensorflow:loss = 44.69645, step = 901 (0.494 sec)
    INFO:tensorflow:Saving checkpoints for 1000 into ongoing/train3/model.ckpt.
    INFO:tensorflow:Loss for final step: 44.18133.





    <tensorflow.python.estimator.canned.linear.LinearClassifier at 0xb21883e80>




```python
model_imp.evaluate(input_fn=get_input_fn(df_test_new, 
                                      num_epochs=1,
                                      n_batch = 128,
                                      shuffle=False),
                                      steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2018-08-29-19:11:05
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ongoing/train3/model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Evaluation [100/1000]
    INFO:tensorflow:Finished evaluation at 2018-08-29-19:11:06
    INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.8358209, accuracy_baseline = 0.76377374, auc = 0.88401634, auc_precision_recall = 0.69599575, average_loss = 0.35122654, global_step = 1000, label/mean = 0.23622628, loss = 44.67437, precision = 0.68986726, prediction/mean = 0.23320661, recall = 0.55408216
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 1000: ongoing/train3/model.ckpt-1000





    {'accuracy': 0.8358209,
     'accuracy_baseline': 0.76377374,
     'auc': 0.88401634,
     'auc_precision_recall': 0.69599575,
     'average_loss': 0.35122654,
     'label/mean': 0.23622628,
     'loss': 44.67437,
     'precision': 0.68986726,
     'prediction/mean': 0.23320661,
     'recall': 0.55408216,
     'global_step': 1000}



The new accuracy level is 83.58 percent. It is four percent higher than
the previous model.

Finally, you can add a regularization term to prevent overfitting.

**Step 5**: Hyperparameter:Lasso & Ridge

Your model can suffer from **overfitting** or **underfitting**.

-   Overfitting: The model is unable to generalize the prediction to new
    data

-   Underfitting: The model is unable to capture the pattern of the
    data. i.e., linear regression when the data is non-linear

When a model has lots of parameters and a relatively low amount of data,
it leads to poor predictions. Imagine, one group only have three
observations; the model will compute a weight for this group. The weight
is used to make a prediction; if the observations of the test set for
this particular group is entirely different from the training set, then
the model will make a wrong prediction. During the evaluation with the
training set, the accuracy is good, but not good with the test set
because the weights computed is not the true one to generalize the
pattern. In this case, it does not make a reasonable prediction on
unseen data.

To prevent overfitting, regularization gives you the possibilities to
control for such complexity and make it more generalizable. There are
two regularization techniques:

-   L1: Lasso

-   L2: Ridge

In TensorFlow, you can add these two hyperparameters in the `optimizer`.
For instance, the higher the hyperparameter `L2`, the weight tends to be
very low and close to zero. The fitted line will be very flat, while an
`L2` close to zero implies the weights are close to the regular linear
regression.

You can try by yourself the different value of the hyperparameters and
see if you can increase the accuracy level.

**Note** that if you change the hyperparameter, you need to delete the
folder `ongoing/train4` otherwise the model will start with the
previously trained model.

Let's see how is the accuracy with the hype


```python
model_regu = tf.estimator.LinearClassifier(
    model_dir="ongoing/train4", feature_columns=categorical_features+base_columns+education_x_occupation+age_buckets_x_education_x_occupation,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.9,
        l2_regularization_strength=5))

model_regu.train(input_fn=get_input_fn(df_train_new, 
                                      num_epochs=None,
                                      n_batch = 128,
                                      shuffle=False),
                                      steps=1000)
```

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_model_dir': 'ongoing/train4', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0xb20aa39e8>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into ongoing/train4/model.ckpt.
    INFO:tensorflow:loss = 88.722855, step = 1
    INFO:tensorflow:global_step/sec: 106.123
    INFO:tensorflow:loss = 50.38778, step = 101 (0.947 sec)
    INFO:tensorflow:global_step/sec: 248.108
    INFO:tensorflow:loss = 55.38014, step = 201 (0.400 sec)
    INFO:tensorflow:global_step/sec: 247.992
    INFO:tensorflow:loss = 46.806694, step = 301 (0.403 sec)
    INFO:tensorflow:global_step/sec: 161.351
    INFO:tensorflow:loss = 38.68271, step = 401 (0.622 sec)
    INFO:tensorflow:global_step/sec: 255.83
    INFO:tensorflow:loss = 56.99398, step = 501 (0.389 sec)
    INFO:tensorflow:global_step/sec: 222.473
    INFO:tensorflow:loss = 33.263622, step = 601 (0.452 sec)
    INFO:tensorflow:global_step/sec: 259.506
    INFO:tensorflow:loss = 37.7902, step = 701 (0.384 sec)
    INFO:tensorflow:global_step/sec: 236.871
    INFO:tensorflow:loss = 61.732605, step = 801 (0.424 sec)
    INFO:tensorflow:global_step/sec: 209.298
    INFO:tensorflow:loss = 46.938225, step = 901 (0.477 sec)
    INFO:tensorflow:Saving checkpoints for 1000 into ongoing/train4/model.ckpt.
    INFO:tensorflow:Loss for final step: 43.4942.





    <tensorflow.python.estimator.canned.linear.LinearClassifier at 0xb20aa3cc0>




```python
model_regu.evaluate(input_fn=get_input_fn(df_test_new, 
                                      num_epochs=1,
                                      n_batch = 128,
                                      shuffle=False),
                                      steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2018-08-29-19:11:18
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ongoing/train4/model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Evaluation [100/1000]
    INFO:tensorflow:Finished evaluation at 2018-08-29-19:11:20
    INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.83833915, accuracy_baseline = 0.76377374, auc = 0.8869794, auc_precision_recall = 0.7014905, average_loss = 0.34691378, global_step = 1000, label/mean = 0.23622628, loss = 44.12581, precision = 0.69720596, prediction/mean = 0.23662092, recall = 0.5579823
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 1000: ongoing/train4/model.ckpt-1000





    {'accuracy': 0.83833915,
     'accuracy_baseline': 0.76377374,
     'auc': 0.8869794,
     'auc_precision_recall': 0.7014905,
     'average_loss': 0.34691378,
     'label/mean': 0.23622628,
     'loss': 44.12581,
     'precision': 0.69720596,
     'prediction/mean': 0.23662092,
     'recall': 0.5579823,
     'global_step': 1000}



With this hyperparameter, you slightly increase the accuracy metrics. In
the next tutorial, you will learn how to improve a linear classifier
using a kernel method.

# Summary

To train a model, you need to:

-   Define the features: Independent variables: X

-   Define the label: Dependent variable: y

-   Construct a train/test set

-   Define the initial weight

-   Define the loss function: MSE

-   Optimize the model: Gradient descent

-   Define:

-   Learning rate

-   Number of epoch

-   Batch size

-   Number of class

In this tutorial, you learned how to use the high-level API for a linear
regression classifier. You need to define:

1.  Feature columns. If continuous:
    `tf.feature_column.numeric_column()`. You can populate a list with
    python list comprehension

2.  The estimator: `tf.estimator.LinearClassifier(feature_columns, model_dir, n_classes = 2)`

3.  A function to import the data, the batch size and epoch: `input_fn()`

After that, you are ready to train, evaluate and make a prediction with
`train()`, `evaluate()` and `predict()`

To improve the performance of the model, you can:

-   Use polynomial regression

-   Interaction term: `tf.feature_column.crossed_column`

-   Add regularization parameter
