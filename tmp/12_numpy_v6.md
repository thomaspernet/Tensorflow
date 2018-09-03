
# What is numpy?


Numpy is an open source library available in Python that aids in
mathematical, scientific, engineering, and data science programming.
Numpy is an incredible library to perform mathematical and statistical
operations. It works perfectly well for multi-dimensional arrays and
matrices multiplication

For any scientific project, numpy is the tool to know. It has been built
to work with the N-dimensional array, linear algebra, random number,
Fourier transform, etc. It can be integrated to C/C++ and Fortran.

Numpy is a programming language that deals with multi-dimensional arrays
and matrices. On top of the arrays and matrices, Numpy supports a large
number of mathematical operations. In this part, we will review the
essential functions that you need to know for the tutorial on
'TensorFlow.'

## Why use numpy?


Numpy is memory efficiency, meaning it can handle the vast amount of
data more accessible than any other library. Besides, numpy is very
convenient to work with, especially for matrix multiplication and
reshaping. On top of that, numpy is fast. In fact, TensorFlow and Scikit
learn to use numpy array to compute the matrix multiplication in the
back end.

## How to install numpy?


To install Pandas library, please refer our tutorial [How to install
TensorFlow](https://www.guru99.com/download-install-tensorflow.html).
Numpy is installed by default. In remote case, numpy not installed-

You can install numpy using:

-   Anaconda: `conda conda install -c anaconda numpy`

-   In Jupyter Notebook :

```
import sys
!conda install --yes --prefix {sys.prefix} numpy
```

### Import Numpy and Check Version

The command to import numpy is


```python
import numpy as np
```

Above code renames the Numpy namespace to `np`. This permits us to prefix Numpy function, methods, and attributes with `np` instead of typing `numpy`. It is the standard shortcut you will find in the numpy literature

To check your installed version of Numpy use the command


```python
print (np.__version__)
```

    1.14.5


### Create a Numpy Array

Simplest way to create an array in Numpy is to use Python List


```python
myPythonList = [1,9,8,3]
```

To convert python list to a numpy array by using the object `np.array`.


```python
numpy_array_from_list = np.array(myPythonList)
```

To display the contents of the list


```python
numpy_array_from_list
```




    array([1, 9, 8, 3])



In practice, there is no need to declare a Python List. The operation
can be combined.


```python
a = np.array([1,9,8,3])
```

**NOTE**: Numpy documentation states use of `np.ndarray` to create
an array. However, this the recommended method

You can also create a numpy array from a Tuple

## Mathematical Operations on an Array

You could perform mathematical operations like additions, subtraction,
division and multiplication on an array. The syntax is the array name
followed by the operation (+.-,*,/) followed by the operand

Example:


```python
numpy_array_from_list + 10
```




    array([11, 19, 18, 13])



This operation adds 10 to each element of the numpy array.

## Shape of Array


You can check the shape of the array with the object `shape` preceded by
the name of the array. In the same way, you can check the type with
`dtypes`.


```python
import numpy as np
a  = np.array([1,2,3])
print(a.shape)
print(a.dtype)
```

    (3,)
    int64


An integer is a value without decimal. If you create an array with
decimal, then the type will change to float.


```python
#### Different type
b  = np.array([1.1,2.0,3.2])
print(b.dtype)
```

    float64


### 2 Dimension Array

You can add a dimension with a ',' coma

Note that it has to be within the bracket `[]`


```python
### 2 dimension
c = np.array([(1,2,3),
              (4,5,6)])
print(c.shape)
```

    (2, 3)


### 3 Dimension Array

Higher dimension can be constructed as follow:


```python
### 3 dimension
d = np.array([
    [[1, 2,3],
        [4, 5, 6]],
    [[7, 8,9],
        [10, 11, 12]]
])
print(d.shape)
```

    (2, 2, 3)


### np.zeros and np.ones


You can create matrix full of zeroes or ones. It can be used when you
initialized the weights during the first iteration in TensorFlow.

The syntax is

```
np.zeros(shape, dtype=float, order='C')

np.ones(shape,dtype=float, order='C')
```

Here,

**Shape**: is the shape of the array

**Dtype***: is the datatype. It is optional. The default value is
float64*

**Order**: Default is C which is an essential row style.

Example:


```python
np.zeros((2,2))
```




    array([[0., 0.],
           [0., 0.]])




```python
## Create 1
np.ones((1,2,3), dtype=np.int16)
```




    array([[[1, 1, 1],
            [1, 1, 1]]], dtype=int16)



### Reshape and Flatten Data


In some occasion, you need to reshape the data from wide to long.


```python
e  = np.array([(1,2,3), (4,5,6)])
print(e)
e.reshape(3,2)
```

    [[1 2 3]
     [4 5 6]]





    array([[1, 2],
           [3, 4],
           [5, 6]])



When you deal with some neural network like convnet, you need to flatten
the array. You can use `flatten()`


```python
e.flatten()
```




    array([1, 2, 3, 4, 5, 6])



### hstack and vstack


Numpy library has also two convenient function to horizontally or
vertically append the data. Lets study them with an example:


```python
## Stack
f = np.array([1,2,3])
g = np.array([4,5,6])

print('Horizontal Append:', np.hstack((f, g)))
print('Vertical Append:', np.vstack((f, g)))
```

    Horizontal Append: [1 2 3 4 5 6]
    Vertical Append: [[1 2 3]
     [4 5 6]]


### Generate Random Numbers


To generate random numbers for Gaussian distribution use

numpy .random.normal(loc, scale, size)

Here

-   Loc: the `mea``n. The center of distribution`

-   `scale: ``standard deviation`.

-   Size: number of returns


```python
## Generate random nmber from normal distribution
normal_array = np.random.normal(5, 0.5, 10)
print(normal_array)
```

    [4.42977249 5.05345355 4.10121374 4.76294476 5.27110877 5.31235603
     4.752106   4.28209053 4.80257037 5.12788267]


### Asarray

Consider the following 2-D matrix with four rows and four columns filled
by 1


```python
A = np.matrix(np.ones((4,4)))
```

If you want to change the value of the matrix, you cannot. The reason
is, it is not possible to change a copy.


```python
np.array(A)[2]=2
print(A)
```

    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]]


`Matrix is immutable`. You can use `asarray` if you want to add
modification in the original array. let’s see if any change occurs when
you want to change the value of the third rows with the value 2


```python
np.asarray(A)[2]=2
print(A)
```

    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [2. 2. 2. 2.]
     [1. 1. 1. 1.]]


Code Explanation: 
`np.asarray(A)`: converts the matrix A to an array

`[2]`: select the third rows

### Arrange


In some occasion, you want to create value evenly spaced within a given
interval. For instance, you want to create values from 1 to 10; you can
use `arrange`

Syntax:

numpy.arange(start, stop,step)

-   Start: Start of interval

-   Stop: End of interval

-   Step: Spacing between values. Default step is 1

Example:


```python
np.arange(1, 11)
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])



If you want to change the step, you can add a third number in the
parenthesis. It will change the step.


```python
np.arange(1, 14, 4)
```




    array([ 1,  5,  9, 13])



### Linspace


Linspace gives evenly spaced samples.

Syntax: numpy.linspace(start, stop, num, endpoint)

Here,

-   Start: Starting value of the sequence

-   Stop: End value of the sequence

-   Num: Number of samples to generate. Default is 50

-   Endpoint: If True (default), stop is the last value. If False, stop
    value is not included.

For instance, it can be used to create 10 values from 1 to 5 evenly
spaced.


```python
np.linspace(1.0, 5.0, num=10)
```




    array([1.        , 1.44444444, 1.88888889, 2.33333333, 2.77777778,
           3.22222222, 3.66666667, 4.11111111, 4.55555556, 5.        ])



If you do not want to include the last digit in the interval, you can
set `endpoint` to false


```python
np.linspace(1.0, 5.0, num=5, endpoint=False)
```




    array([1. , 1.8, 2.6, 3.4, 4.2])



### LogSpace

LogSpace returns even spaced numbers on a log scale. Logspace has the
same parameters as np.linspace.


```python
np.logspace(3.0, 4.0, num=4)
```




    array([ 1000.        ,  2154.43469003,  4641.58883361, 10000.        ])



Finaly, if you want to check the size of an array, you can use
`itemsize`


```python
x = np.array([1,2,3], dtype=np.complex128)
x.itemsize
```




    16



The `x` element has 16 bytes.

### Indexing and slicing

Slicing data is trivial with numpy. We will slice the matrice
`e``. ``Note that, in Python, you need to use the brackets to return the rows or columns`


```python
## Slice

e  = np.array([(1,2,3), (4,5,6)])
print(e)
```

    [[1 2 3]
     [4 5 6]]


Remember with numpy the first array/column starts at 0.


```python
## First column
print('First row:', e[0])

## Second col
print('Second row:', e[1])
```

    First row: [1 2 3]
    Second row: [4 5 6]


In Python, like many other languages,

-   The values before the comma stand for the rows

-   The value on the rights stands for the columns.

-   If you want to select a column, you need to add `:` before the
    column index.

-   `:` means you want all the rows from the selected column.


```python
print('Second column:', e[:,1])
```

    Second column: [2 5]


To return the first two values of the second row. You use : to select
all columns up to the second


```python
## 
print(e[1, :2])
```

    [4 5]


### Statistical function

Numpy is equipped with the robust statistical function as listed below

| Function           | Numpy       |
|--------------------|-------------|
| Min                | np.min()    |
| Max                | np.max()    |
| Mean               | np.mean()   |
| Median             | np.median() |
| Standard deviation | np.stdt()   |


```python
## Statistical function
### Min 
print(np.min(normal_array))

### Max 
print(np.max(normal_array))
### Mean 
print(np.mean(normal_array))
### Median
print(np.median(normal_array))
### Sd
print(np.std(normal_array))
```

    4.101213739969163
    5.31235602845462
    4.789549889424554
    4.782757561802308
    0.39400257302018965


### Dot Product

Numpy is powerful library for matrices computation. For instance, you
can compute the dot product with `np.dot`


```python
## Linear algebra
### Dot product: product of two arrays
f = np.array([1,2])
g = np.array([4,5])
### 1*4+2*5
np.dot(f, g)
```




    14



### Matrix Multiplication

In the same way, you can compute matrices multiplication with
`np.matmul`


```python
### Matmul: matruc product of two arrays
h = [[1,2],[3,4]] 
i = [[5,6],[7,8]] 
### 1*5+2*7 = 19
np.matmul(h, i)
```




    array([[19, 22],
           [43, 50]])



### Determinant

Last but not least, if you need to compute the determinant, you can use
`np.linalg.det()`. Note that numpy takes care of the dimension.


```python
## Determinant 2*2 matrix
### 5*8-7*6
np.linalg.det(i)
```




    -2.000000000000005



## Summary


Below, a summary of the essential functions used with numpy

| Objective             | Code             |
|-----------------------|------------------|
| Create array          | array([1,2,3])   |
| print the shape       | array([.]).shape |
| reshape               | reshape          |
| flat an array         | flatten          |
| append vertically     | vstack           |
| append horizontally   | hstack           |
| create a matrix       | matrix           |
| create space          | arrange          |
| Create a linear space | linspace         |
| Create a log space    | logspace         |


Below is a summary of basic statistical and arithmetical function

| Objective          | Code     |
|--------------------|----------|
| min                | min()    |
| max                | max()    |
| mean               | mean()   |
| median             | median() |
| standard deviation | std()    |

Code:

import numpy as np



##Create array

### list


```python
myPythonList = [1,9,8,3]

numpy_array_from_list = np.array(myPythonList)
```

### Directly in numpy


```python
np.array([1,9,8,3])


```




    array([1, 9, 8, 3])



### Shape


```python
a = np.array([1,2,3])
print(a.shape)


```

    (3,)


### Type


```python
print(a.dtype)


```

    int64


### 2D array


```python
c = np.array([(1,2,3), (4,5,6)])


```

### 3D array


```python
d = np.array([ 
[[1, 2,3],
[4, 5, 6]],
[[7, 8,9],
[10, 11, 12]]
])


```

### Reshape


```python
e = np.array([(1,2,3), (4,5,6)])

print(e)

e.reshape(3,2)


```

    [[1 2 3]
     [4 5 6]]





    array([[1, 2],
           [3, 4],
           [5, 6]])



### Flatten


```python
e.flatten()


```




    array([1, 2, 3, 4, 5, 6])



### hstack & vstack


```python
f = np.array([1,2,3])

g = np.array([4,5,6])



np.hstack((f, g))

np.vstack((f, g))


```




    array([[1, 2, 3],
           [4, 5, 6]])



### random number


```python
normal_array = np.random.normal(5, 0.5, 10)


```

### asarray


```python
A = np.matrix(np.ones((4,4)))

np.asarray(A)


```




    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]])



### Arrange


```python
np.arange(1, 11)


```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])



### linspace


```python
np.linspace(1.0, 5.0, num=10)


```




    array([1.        , 1.44444444, 1.88888889, 2.33333333, 2.77777778,
           3.22222222, 3.66666667, 4.11111111, 4.55555556, 5.        ])



### logspace


```python
np.logspace(3.0, 4.0, num=4)


```




    array([ 1000.        ,  2154.43469003,  4641.58883361, 10000.        ])



### Slicing

#### rows


```python
e = np.array([(1,2,3), (4,5,6)])

e[0]


```




    array([1, 2, 3])



#### columns


```python
e[:,1]


```




    array([2, 5])



#### rows and columns


```python
e[1, :2]
```




    array([4, 5])


