---
title: Deep Learning Libraries?
author: Thomas
date: '2018-08-29'
slug: deep-learnin-libraires
categories: []
tags:
  - AI
header:
  caption: ''
  image: ''
---  

<style>
body {
text-align: justify}
</style>

# Deep learning libraries

Artificial intelligence is growing in popularity since 2016 with, 20% of
the big companies using AI in their businesses (McKinsey
[report](https://www.mckinsey.com/business-functions/mckinsey-analytics/our-insights/crossing-the-frontier-how-to-apply-ai-for-impact),
2018). As per the same report AI can create substantial value across
industries. In banking, for instance, the potential of AI is estimated
at **$** 300 billion, in retail the number skyrocket to **$** 600
billion.

To unlock the potential value of AI, companies must choose the right
deep learning framework. In this tutorial, you will learn about the
different libraries available to carry out deep learning tasks. Some
libraries have been around for years while new library like TensorFlow
has come to light in recent years.

## Torch

Torch is an old open source machine learning library. It is first
released was 15 years ago. It is primary programming languages is LUA,
but has an implementation in C. Torch supports a vast library for
machine learning algorithms, including deep learning. It supports CUDA
implementation for parallel computation.

Torch is used by most of the leading labs such as Facebook, Google,
Twitter, Nvidia, and so on. Torch has a library in Python names Pytorch.

## Infer.net

Infer.net is developed and maintained by Microsoft. Infer.net is a
library with a primary focus on the Bayesian statistic. Infer.net is
designed to offer practitioners state-of-the-art algorithms for
probabilistic modeling. The library contains analytical tools such as
Bayesian analysis, hidden Markov chain, clustering.

## Keras

Keras is a Python framework for deep learning. It is a convenient
library to construct any deep learning algorithm. The advantage of Keras
is that it uses the same Python code to run on CPU or GPU. Besides, the
coding environment is pure and allows for training state-of-the-art
algorithm for computer vision, text recognition among other.

Keras has been developed by François Chollet, a researcher at Google.
Keras is used in prominent organizations like CERN, Yelp, Square or
Google, Netflix, and Uber.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/5_Deep_learning_libraries_v9_files/image001.png)

## Theano

Theano is deep learning library developed by the Université de Montréal
in 2007. It offers fast computation and can be run on both CPU and GPU.
Theano has been developed to train deep neural network algorithms.

## MICROSOFT COGNITIVE TOOLKIT

Microsoft toolkit, previously know as CNTK, is a deep learning library
developed by Microsoft. According to Microsoft, the library is among the
fastest on the market. Microsoft toolkit is an open-source library,
although Microsoft is using it extensively for its product like Skype,
Cortana, Bing, and Xbox. The toolkit is available both in Python and
C++.

## MXNet

MXnet is a recent deep learning library. It is accessible with multiple
programming languages including C++, Julia, Python and R. MXNet can be
configured to work on both CPU and GPU. MXNet includes state-of-the-art
deep learning architecture such as Convolutional Neural Network and Long
Short-Term Memory. MXNet is build to work in harmony with dynamic cloud
infrastructure. The main user of MXNet is Amazon

## Caffe

Caffe is a library built by Yangqing Jia when he was a PhD student at
Berkeley. Caffe is written in C++ and can perform computation on both
CPU and GPU. The primary uses of Caffe is Convolutional Neural Network.
Although, In 2017, Facebook extended Caffe with more deep learning
architecture, including Recurrent Neural Network. caffe is used by
academics and startups but also some large companies like Yahoo!.

## TensorFlow

TensorFlow is Google open source project. TensorFlow is the most famous
deep learning library these days. It was released to the public in late
2015

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/5_Deep_learning_libraries_v9_files/image002.png)

TensorFlow is developed in C++ and has convenient Python API, although
C++ APIs are also available. Prominent companies like Airbus, Google,
IBM and so on are using TensorFlow to produce deep learning algorithms.

| Library                     | Platform                                        | Written in        | Cuda support | Parallel Execution | Has trained models | RNN | CNN |
|-----------------------------|-------------------------------------------------|-------------------|--------------|--------------------|--------------------|-----|-----|
| Torch                       | Linux, MacOS, Windows                           | Lua               | Yes          | Yes                | Yes                | Yes | Yes |
| Infer.Net                   | Linux, MacOS, Windows                           | Visual Studio     | No           | No                 | No                 | No  | No  |
| Keras                       | Linux, MacOS, Windows                           | Python            | Yes          | Yes                | Yes                | Yes | Yes |
| Theano                      | Cross-platform                                  | Python            | Yes          | Yes                | Yes                | Yes | Yes |
| TensorFlow                  | Linux, MacOS, Windows, Android                  | C++, Python, CUDA | Yes          | Yes                | Yes                | Yes | Yes |
| MICROSOFT COGNITIVE TOOLKIT | Linux, Windows, Mac with Docker                 | C++               | Yes          | Yes                | Yes                | Yes | Yes |
|                             |                                                 |                   |              |                    |                    |     |     |
| Caffe                       | Linux, MacOS, Windows                           | C++               | Yes          | Yes                | Yes                | Yes | Yes |
|                             |                                                 |                   |              |                    |                    |     |     |
| MXNet                       | Linux, Windows, MacOs, Android, iOS, Javascript | C++               | Yes          | Yes                | Yes                | Yes | Yes |

**Verdict:**

TensorFlow is the best library of all because it is built to be
accessible for everyone. Tensorflow library incorporates different API
to built at scale deep learning architecture like CNN or RNN. TensorFlow
is based on graph computation, it allows the developer to visualize the
construction of the neural network with Tensorboad. This tool is helpful
to debug the program. Finally, Tensorflow is built to be deployed at
scale. It runs on CPU and GPU.

Tensorflow attracts the largest popularity on GitHub compare to the
other deep learning framework.

### Google Cloud ML

Google provides for developer pre-trained model available in Cloud
AutoML. This solution exists for a developer without a strong background
in machine learning. Developers can use state-of-the-art Google's
pre-trained model on their data. It allows any developers to train and
evaluate any model in just a few minutes.

Google currently provides a REST API for computer vision, speech
recognition, translation, and NLP.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/5_Deep_learning_libraries_v9_files/image003.jpg)

Using Google Cloud, you can train a machine learning framework build on
TensorFlow, Scikit-learn, XGBoost or Keras. Google Cloud machine
learning will train the models across its cloud.

The advantage to use Google cloud computing is the simplicity to deploy
machine learning into production. There is no need to set up Docker
container. Besides, the cloud takes care of the infrastructure. It knows
how to allocate resources with CPUs, GPUs, and TPUs. It makes the
training faster with paralleled computation.

### AWS SageMaker

A major competitor to Google Cloud is Amazon cloud, AWS. Amazon has
developed Amazon SageMaker to allow data scientists and developers to
build, train and bring into production any machine learning models.

SageMaker is available in a Jupyter Notebook and includes the most used
machine learning library, TensorFlow, MXNet, Scikit-learn amongst
others. Programs written with SageMaker are automatically run in the
Docker containers. Amazon handles the resource allocation to optimize
the training and deployment.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/5_Deep_learning_libraries_v9_files/image004.png)

Amazon provides API to the developers in order to add intelligence to
their applications. In some occasion, there is no need to
reinventing-the-wheel by building from scratch new models while there
are powerful pre-trained models in the cloud. Amazon provides API
services for computer vision, conversational chatbots and language
services:

The three major available API are:

-   Amazon Rekognition: provides image and video recognition to an app

-   Amazon Comprehend: Perform text mining and neural language
    processing to, for instance, automatize the process of checking the
    legality of financial document

-   Amazon Lex: Add chatbot to an app

### Azure Machine Learning Studio

Probably one of the friendliest approaches to machine learning is Azure
Machine Learning Studio. The significant advantage of this solution is
that no prior programming knowledge is required.

Microsoft Azure Machine Learning Studio is a drag-and-drop collaborative
tool to create, train, evaluate and deploy machine learning solution.
The model can be efficiently deployed as web services and used in
several apps like Excel.

Azure Machine learning interface is interactive, allowing the user to
build a model just by dragging and dropping elements quickly.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/5_Deep_learning_libraries_v9_files/image005.JPG)

When the model is ready, the developer can save it and push it to Azure
[Gallery](https://gallery.azure.ai/solutions) or Azure Marketplace.

Azure Machine learning can be integrated into R or Python their custom
built-in package.

### IBM Watson ML

Watson studio can simplify the data projects with a streamlined process
that allows extracting value and insights from the data to help the
business to get smarter and faster. Watson studio delivers an
easy-to-use collaborative data science and machine learning environment
for building and training models, preparing and analyzing data, and
sharing insights all in one place. Watson Studio is easy to use with a
drag-and-drop code.

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/5_Deep_learning_libraries_v9_files/image006.png)

Watson studio supports some of the most popular frameworks like
Tensorflow, Keras, Pytorch, Caffe and can deploy a deep learning
algorithm on to the latest GPUs from Nvidia to help accelerate modeling.

### Verdict

In our point of view, Google cloud solution is the one that is the most recommended. Google cloud solution provides lower prices the AWS by at least 30% for data storage and machine learning solution. Google is doing an excellent job to democratize AI. It has developed an open source language, TensorFlow, optimized data warehouse connection, provides tremendous tools from data visualization, data analysis to machine learning. Besides, Google Console is ergonomic and much more comprehensive than AWS or Windows.
