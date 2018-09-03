# What is Machine Learning? 

A system that can learn from example through self-improvement and
without being explicitly programmer is known as a machine learning. The
breakthrough comes with the idea that a machine can singularly learn
from the data (i.e., example) to produce accurate results.

Machine learning combines data with statistical tools to predict an
output. This output is then used by corporate to makes actionable
insights. Machine learning is closely related to data mining and
Bayesian predictive modeling. The machine receives data as input, use an
algorithm to formulate answers.

A typical machine learning tasks are to provide a recommendation. For
those who have a Netflix account, all recommendations of movies or
series are based on the user’s historical data. Tech companies are using
unsupervised learning to improve the user experience with personalizing
recommendation.

Machine learning is also used for a variety of task like fraud
detection, predictive maintenance, portfolio optimization, automatize
task and so on.

## Machine Learning vs. Traditional Programming


Traditional programming differs significantly from machine learning. In
traditional programming, a programmer code all the rules in consultation
with an expert in the industry for which software is being developed.
Each rule is based on a logical foundation; the machine will execute an
output following the logical statement. When the system grows complex,
more rules need to be written. It can quickly become unsustainable to
maintain.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/2_What_is_Machine_Learning_v8_files/image001.png)


Machine learning is supposed to overcome this issue. The machine learns
how the input and output data are correlated and it writes a rule. The
programmers do not need to write new rules each time there is new data.
The algorithms adapt in response to new data and experiences to improve
efficacy over time.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/2_What_is_Machine_Learning_v8_files/image002.png)


## How does machine learning work? 

Machine learning is the brain where all the learning takes place. The
way the machine learns is similar to the human being. Humans learn from
experience. The more we know, the more easily we can predict. By
analogy, when we face an unknown situation, the likelihood of success is
lower than the known situation. Machines are trained the same. To make
an accurate prediction, the machine sees an example. When we give the
machine a similar example, it can figure out the outcome. However, like
a human, if its feed a previously unseen example, the machine has
difficulties to predict.

The core objective of machine learning is the **learning** and
**inference**. First of all, the machine learns through the discovery of
patterns. This discovery is made thanks to the **data**. One crucial
part of the data scientist is to choose carefully which data to provide
to the machine. The list of attributes used to solve a problem is called
a **feature vector.** You can think of a feature vector as a subset of
data that is used to tackle a problem.

The machine uses some fancy algorithms to simplify the reality and
transform this discovery into a **model**. Therefore, the learning stage
is used to describe the data and summarize it into a model.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/2_What_is_Machine_Learning_v8_files/image003.png)

For instance, the machine is trying to understand the relationship
between the wage of an individual and the likelihood to go to a fancy
restaurant. It turns out the machine finds a positive relationship
between wage and going to a high-end restaurant: This is the model

**Inferring**

When the model is built, it is possible to test how powerful it is on
never-seen-before data. The new data are transformed into a features
vector, go through the model and give a prediction. This is all the
beautiful part of machine learning. There is no need to update the rules
or train again the model. You can use the model previously trained to
make inference on new data.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/2_What_is_Machine_Learning_v8_files/image004.png)

The life of Machine Learning programs is straightforward and can be
summarized in the following points:

1.  Define a question

2.  Collect data

3.  Visualize data

4.  Train algorithm

5.  Test the Algorithm

6.  Collect feedback

7.  Refine the algorithm

8.  Loop 4-7 until the results are satisfying

9.  Use the model to make a prediction

Once the algorithm gets good at drawing the right conclusions, it
applies that knowledge to new sets of data.

## Machine learning Algorithms and where they are used?


![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/2_What_is_Machine_Learning_v8_files/image005.png)

Machine learning can be grouped into two broad learning tasks:
Supervised and Unsupervised. There are many other algorithms

**Supervised learning**

An algorithm uses training data and feedback from humans to learn the
relationship of given inputs to a given output. For instance, a
practitioner can use marketing expense and weather forecast as input
data to predict the sales of cans.

You can use supervised learning when the output data is known. The
algorithm will predict new data.



There are two categories of supervised learning:

-   Classification task

-   Regression task

**Classification**

Imagine you want to predict the gender of a customer for a commercial.
You will start gathering data on the height, weight, job, salary,
purchasing basket, etc. from your customer database. You know the gender
of each of your customer, it can only be male or female. The objective
of the classifier will be to assign a probability of being a male or a
female (i.e., the label) based on the information (i.e., features you
have collected). When the model learned how to recognize male or female,
you can use new data to make a prediction. For instance, you just got
new information from an unknown customer, and you want to know if it is
a male or female. If the classifier predicts male = 70%, it means the
algorithm is sure at 70% that this customer is a male, and 30% it is a
female.

The label can be of two or more classes. The above example has only two
classes, but if a classifier needs to predict object, it has dozens of
classes (e.g., glass, table, shoes, etc. each object represents a class)

**Regression**

When the output is a continuous value, the task is a regression. For
instance, a financial analyst may need to forecast the value of a stock
based on a range of feature like equity, previous stock performances,
macroeconomics index. The system will be trained to estimate the price
of the stocks with the lowest possible error.

| Algorithm               | Description                                                                                                                                                                                                                                                                                                                                                                                             | Type                                         |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| Linear regression       | Finds a way to correlate each feature to the output to help predict future values.                                                                                                                                                                                                                                                                                                                      | Regression                                   |
| Logistic regression     | Extension of linear regression that’s used for classification tasks. The output variable 3is binary (e.g., only black or white) rather than continuous (e.g., an infinite list of potential colors)                                                                                                                                                                                                     | Classification                               |
| Decision tree           | Highly interpretable classification or regression model that splits data-feature values into branches at decision nodes (e.g., if a feature is a color, each possible color becomes a new branch) until a final decision output is made                                                                                                                                                                 | Regression, Classification                   |
| Naive Bayes             | The Bayesian method is a classification method that makes use of the Bayesian theorem. The theorem updates the prior knowledge of an event with the independent probability of each feature that can affect the event.                                                                                                                                                                                  | Regression, Classification                   |
| Support vector machine  | Support Vector Machine, or SVM, is typically used for the classification task. SVM algorithm finds a hyperplane that optimally divided the classes. It is best used with a non-linear solver.                                                                                                                                                                                                           | Regression (not very common), classification |
| Random forest           | The algorithm is built upon a decision tree to improve the accuracy drastically. Random forest generates many times simple decision trees and uses the 'majority vote' method to decide on which label to return. For the classification task, the final prediction will be the one with the most vote; while for the regression task, the average prediction of all the trees is the final prediction. | Regression, classification                   |
| AdaBoost                | Classification or regression technique that uses a multitude of models to come up with a decision but weighs them based on their accuracy in predicting the outcome                                                                                                                                                                                                                                     | Regression, classification                   |
| Gradient-boosting trees | Gradient-boosting trees is a state-of-the-art classification/regression technique. It is focusing on the error committed by the previous trees and tries to correct it.                                                                                                                                                                                                                                 | Regression, classification                   |                 

**Unsupervised learning**

In unsupervised learning, an algorithm explores input data without being
given an explicit output variable (e.g., explores customer demographic
data to identify patterns)

You can use it when you do not know how to classify the data, and you
want the algorithm to find patterns and classify the data for you

| Algorithm               | Description                                                                                                                                        | Type                |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| K-means clustering      | Puts data into some groups (k) that each contains data with similar characteristics (as determined by the model, not in advance by humans)         | Clustering          |
| Gaussian mixture model  | A generalization of k-means clustering that provides more flexibility in the size and shape of groups (clusters                                    | Clustering          |
| Hierarchical clustering | Splits clusters along a hierarchical tree to form a classification system. Can be used for Cluster loyalty-card customer                           | Clustering          |
| Recommender system      | Help to define the relevant data for making a recommendation.                                                                                      | Clustering          |
| PCA/T-SNE               | Mostly used to decrease the dimensionality of the data. The algorithms reduce the number of features to 3 or 4 vectors with the highest variances. | Dimension Reduction |

### Which Machine Learning Algorithm to select? 


There are plenty of machine learning algorithms. The choice of the
algorithm is based on the objective.

In the example below, the task is to predict the type of flower among
the three varieties. The predictions are based on the length and the
width of the petal. The picture depicts the results of ten different
algorithms. The picture on the top left is the dataset. The data is
classified into three categories: red, light blue and dark blue. There
are some groupings. For instance, from the second image, everything in
the upper left belongs to the red category, in the middle part, there is
a mixture of uncertainty and light blue while the bottom corresponds to
the dark category. The other images show different algorithms and how
they try to classified the data.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/2_What_is_Machine_Learning_v8_files/image006.png)

### Challenges and Limitations of Machine learning  

The primary challenge of machine learning is the lack of data or the
diversity in the dataset. A machine cannot learn if there is no data
available. Besides, a dataset with a lack of diversity gives the machine
a hard time. A machine needs to have heterogeneity to learn meaningful
insight. It is rare that an algorithm can extract information when there
are no or few variations. It is recommended to have at least 20
observations per group to help the machine learn. This constraint leads
to poor evaluation and prediction.

## Application of Machine learning  

**Augmentation**:

-   Machine learning, which assists humans with their day-to-day tasks,
    personally or commercially without having complete control of the
    output. Such machine learning is used in different ways such as
    Virtual Assistant, Data analysis, software solutions. The primary
    user is to reduce errors due to human bias.

**Automation**:

-   Machine learning, which works entirely autonomously in any field
    without the need for any human intervention. For example, robots
    performing the essential process steps in manufacturing plants.

**Finance Industry**

-   Machine learning is growing in popularity in the finance industry.
    Banks are mainly using ML to find patterns inside the data but also
    to prevent fraud.

**Government organization**

-   The government makes use of ML to manage public safety and
    utilities. Take the example of China with the massive face
    recognition. The government uses Artificial intelligence to prevent
    jaywalker.

**Healthcare industry**

-   Healthcare was one of the first industry to use machine learning
    with image detection.

**Marketing**

-   Broad use of AI is done in marketing thanks to abundant access to
    data. Before the age of mass data, researchers develop advanced
    mathematical tools like Bayesian analysis to estimate the value of a
    customer. With the boom of data, marketing department relies on AI
    to optimize the customer relationship and marketing campaign.

**Example of application of Machine Learning in Supply Chain**

Machine learning gives terrific results for visual pattern recognition,
opening up many potential applications in physical inspection and
maintenance across the entire supply chain network.

Unsupervised learning can quickly search for comparable patterns in the
diverse dataset. In turn, the machine can perform quality inspection
throughout the logistics hub, shipment with damage and wear.

For instance, IBM's Watson platform can determine shipping container
damage. Watson combines visual and systems-based data to track, report
and make recommendations in real-time.

In past year stock manager relies extensively on the primary method to
evaluate and forecast the inventory. When combining big data and machine
learning, better forecasting techniques have been implemented (an
improvement of 20 to 30 % over traditional forecasting tools). In term
of sales, it means an increase of 2 to 3 % due to the potential
reduction in inventory costs.

**Example of Machine Learning Google Car**

For example, everybody knows the Google car. The car is full of lasers
on the roof which are telling it where it is regarding the surrounding
area. It has radar in the front, which is informing the car of the speed
and motion of all the cars around it. It uses all of that data to figure
out not only how to drive the car but also to figure out and predict
what potential drivers around the car are going to do. What's impressive
is that the car is processing almost a gigabyte a second of data.

## Why is Machine Learning important? 

Machine learning is the best tool so far to analyze, understand and
identify a pattern in the data. One of the main ideas behind machine
learning is that the computer can be trained to automate tasks that
would be exhaustive or impossible for a human being. The clear breach
from the traditional analysis is that machine learning can take
decisions with minimal human intervention.

Take the following example; a retail agent can estimate the price of a
house based on his own experience and his knowledge of the market.

A machine can be trained to translate the knowledge of an expert into
*features*. The features are all the characteristics of a house,
neighborhood, economic environment, etc. that make the price difference.
For the expert, it took him probably some years to master the art of
estimate the price of a house. His expertise is getting better and
better after each sale.

For the machine, it takes millions of data, (i.e., example) to master
this art. At the very beginning of its learning, the machine makes a
mistake, somehow like the junior salesman. Once the machine sees all the
example, it got enough knowledge to make its estimation. At the same
time, with incredible accuracy. The machine is also able to adjust its
mistake accordingly.

Most of the big company have understood the value of machine learning
and holding data. McKinsey have estimated that the value of analytics
ranges from **$**9.5 trillion to **$**15.4 trillion while **$**5 to 7
trillion can be attributed to the most advanced AI techniques.
