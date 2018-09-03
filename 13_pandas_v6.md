---
title: Pandas library
author: Thomas
date: '2018-08-29'
slug: pandas
categories: []
tags:
  - tf-install
header:
    caption: ''
    image: ''
---

<style>
body {
text-align: justify}
</style>

# What is Pandas?


Pandas is an opensource library that allows to you perform data
manipulation in Python. Pandas library is built on top of Numpy, meaning
Pandas needs Numpy to operate. Pandas provide an easy way to create,
manipulate and wrangle the data. Pandas is also an elegant solution for
time series data.

## Why use Pandas?


Data scientists use Pandas for its following advantages:

-   Easily handles missing data

-   It uses **Series for one-dimensional data structure** and
    **DataFrame for multi-dimensional data structure**

-   It provides an efficient way to slice the data

-   It provides a flexible way to merge, concatenate or reshape the data

-   It includes a powerful time series tool to work with

In a nutshell, Pandas is a useful library in data analysis. It can be
used to perform data manipulation and analysis. Pandas provide powerful
and easy-to-use data structures, as well as the means to quickly perform
operations on these structures.

## How to install Pandas?


To install Pandas library, please refer our tutorial How to install
TensorFlow. Pandas is installed by default. In remote case, pandas not
installed-

You can install Pandas using:

-   Anaconda: `conda install -c anaconda pandas`

-   In Jupyter Notebook :

```
import sys
!conda install --yes --prefix {sys.prefix} pandas
```

## What is a data frame?


A data frame is a two-dimensional array, with labeled axes (rows and
columns). A data frame is a standard way to store data.

Data frame is well-known by statistician and other data practitioners. A
data frame is a tabular data, with rows to store the information and
columns to name the information. For instance, the price can be the name
of a column and 2,3,4 the price values.

Below a picture of a Pandas data frame:

![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/13_pandas_v6_files/image001.png)


## What is a Series?


A series is a one-dimensional data structure. It
can have any data structure like integer, float, and string. It is
useful when you want to perform computation or return a one-dimensional
array. A series, by definition, cannot have multiple columns. For the
latter case, please use the data frame structure.

Series has one parameters:

- *Data: can be a list, dictionary or scalar value*


```python
import pandas as pd
pd.Series([1., 2., 3.])
```




    0    1.0
    1    2.0
    2    3.0
    dtype: float64



You can add the index with `index`. It helps to name the rows. The
length should be equal to the size of the column


```python
pd.Series([1., 2., 3.], index=['a', 'b', 'c'])
```




    a    1.0
    b    2.0
    c    3.0
    dtype: float64



Below, you create a Pandas series with a missing value for the third rows. Note, missing values in Python are noted "NaN." You can use numpy to create missing value: np.nan artificially


```python
import numpy as np
pd.Series([1,2,np.nan])
```




    0    1.0
    1    2.0
    2    NaN
    dtype: float64



### Create Data frame

You can convert a numpy array to a pandas data frame with
`pd.Data frame()`. The opposite is also possible. To convert a pandas
Data Frame to an array, you can use `np.array()`


```python
## Numpy to pandas
import numpy as np
h = [[1,2],[3,4]] 
df_h = pd.DataFrame(h)
print('Data Frame:', df_h)

## Pandas to numpy
df_h_n = np.array(df_h)
print('Numpy array:', df_h_n)
```

    Data Frame:    0  1
    0  1  2
    1  3  4
    Numpy array: [[1 2]
     [3 4]]


You can also use a dictionary to create a Pandas dataframe.


```python
dic = {'Name': ["John", "Smith"], 'Age': [30, 40]}

pd.DataFrame(data=dic)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smith</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



### Range Data

Pandas have a convenient API to create a range of date

`pd.data_range(``date,period,frequency``)`:

-   The first parameter is the starting date

-   The second parameter is the number of periods (optional if the end
    date is specified)

-   The last parameter is the frequency: day: 'D,' month: 'M' and year:
    'Y.'


```python
## Create date

# Days

dates_d = pd.date_range('20300101', periods=6, freq='D')

print('Day:', dates_d)
```

    Day: DatetimeIndex(['2030-01-01', '2030-01-02', '2030-01-03', '2030-01-04',
                   '2030-01-05', '2030-01-06'],
                  dtype='datetime64[ns]', freq='D')


We month frequency


```python
# Months
dates_m = pd.date_range('20300101', periods=6, freq='M')
print('Month:', dates_m)
```

    Month: DatetimeIndex(['2030-01-31', '2030-02-28', '2030-03-31', '2030-04-30',
                   '2030-05-31', '2030-06-30'],
                  dtype='datetime64[ns]', freq='M')


### Inspecting data

You can check the head or tail of the dataset with `head()`, or `tail()`
preceded by the name of the panda's data frame

**Step 1**: Create a random sequence with numpy. The sequence has 4 columns
and 6 rows


```python
random = np.random.randn(6,4)
```



**Step 2**: Then you create a data frame using pandas.

Use `dates_m` as an index for the data frame. It means each row will be
given a `name` or an index, corresponding to a date.

Finally, you give a name to the 4 columns with the argument columns


```python
#Create data with date
df = pd.DataFrame(random,
                      index=dates_m,
                      columns=list('ABCD'))
```

**Step 3**: Using head function


```python
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2030-01-31</th>
      <td>1.416223</td>
      <td>0.514896</td>
      <td>-0.641868</td>
      <td>-1.405242</td>
    </tr>
    <tr>
      <th>2030-02-28</th>
      <td>1.018193</td>
      <td>0.175573</td>
      <td>-0.082217</td>
      <td>-0.664702</td>
    </tr>
    <tr>
      <th>2030-03-31</th>
      <td>1.971421</td>
      <td>0.875303</td>
      <td>0.550543</td>
      <td>0.470444</td>
    </tr>
  </tbody>
</table>
</div>



**Step 4**: Using tail function


```python
df.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2030-04-30</th>
      <td>-0.270187</td>
      <td>0.639563</td>
      <td>0.688070</td>
      <td>1.646974</td>
    </tr>
    <tr>
      <th>2030-05-31</th>
      <td>1.019391</td>
      <td>0.695651</td>
      <td>0.434331</td>
      <td>-1.090715</td>
    </tr>
    <tr>
      <th>2030-06-30</th>
      <td>-0.209629</td>
      <td>-0.932602</td>
      <td>-0.736949</td>
      <td>-0.652596</td>
    </tr>
  </tbody>
</table>
</div>



**Step 5**: An excellent practice to get a clue about the data is to use
`describe()`. It provides the counts, mean, std, min, max and percentile
of the dataset.


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.824235</td>
      <td>0.328064</td>
      <td>0.035318</td>
      <td>-0.282639</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.895414</td>
      <td>0.660161</td>
      <td>0.619604</td>
      <td>1.139000</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.270187</td>
      <td>-0.932602</td>
      <td>-0.736949</td>
      <td>-1.405242</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.097326</td>
      <td>0.260404</td>
      <td>-0.501955</td>
      <td>-0.984212</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.018792</td>
      <td>0.577230</td>
      <td>0.176057</td>
      <td>-0.658649</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.317015</td>
      <td>0.681629</td>
      <td>0.521490</td>
      <td>0.189684</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.971421</td>
      <td>0.875303</td>
      <td>0.688070</td>
      <td>1.646974</td>
    </tr>
  </tbody>
</table>
</div>



### Slice data

The last point of this tutorial is about how to slice a pandas data
frame.

You can use the column name to extract data in a particular column.


```python
## Slice
### Using name
df['A']
```




    2030-01-31    1.416223
    2030-02-28    1.018193
    2030-03-31    1.971421
    2030-04-30   -0.270187
    2030-05-31    1.019391
    2030-06-30   -0.209629
    Freq: M, Name: A, dtype: float64



To select multiple columns, you need to use two times the bracket, [[..,..]]

The first pair of bracket means you want to select columns, the second
pairs of bracket tells what columns you want to return.


```python
df[['A', 'B']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2030-01-31</th>
      <td>1.416223</td>
      <td>0.514896</td>
    </tr>
    <tr>
      <th>2030-02-28</th>
      <td>1.018193</td>
      <td>0.175573</td>
    </tr>
    <tr>
      <th>2030-03-31</th>
      <td>1.971421</td>
      <td>0.875303</td>
    </tr>
    <tr>
      <th>2030-04-30</th>
      <td>-0.270187</td>
      <td>0.639563</td>
    </tr>
    <tr>
      <th>2030-05-31</th>
      <td>1.019391</td>
      <td>0.695651</td>
    </tr>
    <tr>
      <th>2030-06-30</th>
      <td>-0.209629</td>
      <td>-0.932602</td>
    </tr>
  </tbody>
</table>
</div>



You can slice the rows with :

The code below returns the first three rows


```python
### using a slice for row
df[0:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2030-01-31</th>
      <td>1.416223</td>
      <td>0.514896</td>
      <td>-0.641868</td>
      <td>-1.405242</td>
    </tr>
    <tr>
      <th>2030-02-28</th>
      <td>1.018193</td>
      <td>0.175573</td>
      <td>-0.082217</td>
      <td>-0.664702</td>
    </tr>
    <tr>
      <th>2030-03-31</th>
      <td>1.971421</td>
      <td>0.875303</td>
      <td>0.550543</td>
      <td>0.470444</td>
    </tr>
  </tbody>
</table>
</div>



The loc function is used to select columns by names. As usual, the
values before the coma stand for the rows and after refer to the column.
You need to use the brackets to select more than one column.


```python
## Multi col
df.loc[:,['A','B']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2030-01-31</th>
      <td>1.416223</td>
      <td>0.514896</td>
    </tr>
    <tr>
      <th>2030-02-28</th>
      <td>1.018193</td>
      <td>0.175573</td>
    </tr>
    <tr>
      <th>2030-03-31</th>
      <td>1.971421</td>
      <td>0.875303</td>
    </tr>
    <tr>
      <th>2030-04-30</th>
      <td>-0.270187</td>
      <td>0.639563</td>
    </tr>
    <tr>
      <th>2030-05-31</th>
      <td>1.019391</td>
      <td>0.695651</td>
    </tr>
    <tr>
      <th>2030-06-30</th>
      <td>-0.209629</td>
      <td>-0.932602</td>
    </tr>
  </tbody>
</table>
</div>



There is another method to select multiple rows and columns in Pandas.
You can use iloc[]. This method uses the index instead of the columns
name. The code below returns the same data frame as above


```python
df.iloc[:, :2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2030-01-31</th>
      <td>1.416223</td>
      <td>0.514896</td>
    </tr>
    <tr>
      <th>2030-02-28</th>
      <td>1.018193</td>
      <td>0.175573</td>
    </tr>
    <tr>
      <th>2030-03-31</th>
      <td>1.971421</td>
      <td>0.875303</td>
    </tr>
    <tr>
      <th>2030-04-30</th>
      <td>-0.270187</td>
      <td>0.639563</td>
    </tr>
    <tr>
      <th>2030-05-31</th>
      <td>1.019391</td>
      <td>0.695651</td>
    </tr>
    <tr>
      <th>2030-06-30</th>
      <td>-0.209629</td>
      <td>-0.932602</td>
    </tr>
  </tbody>
</table>
</div>



### Drop a column

You can drop columns using `pd.drop()`


```python
df.drop(columns=['A', 'C'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2030-01-31</th>
      <td>0.514896</td>
      <td>-1.405242</td>
    </tr>
    <tr>
      <th>2030-02-28</th>
      <td>0.175573</td>
      <td>-0.664702</td>
    </tr>
    <tr>
      <th>2030-03-31</th>
      <td>0.875303</td>
      <td>0.470444</td>
    </tr>
    <tr>
      <th>2030-04-30</th>
      <td>0.639563</td>
      <td>1.646974</td>
    </tr>
    <tr>
      <th>2030-05-31</th>
      <td>0.695651</td>
      <td>-1.090715</td>
    </tr>
    <tr>
      <th>2030-06-30</th>
      <td>-0.932602</td>
      <td>-0.652596</td>
    </tr>
  </tbody>
</table>
</div>



### Concatenation


You can concatenate two DataFrame in Pandas. You can use pd.concat()

First of all, you need to create two DataFrames. So far so good, you are
already familiar with dataframe creation

import numpy as np


```python
df1 = pd.DataFrame({'name': ['John', 'Smith','Paul'],
                    'Age': ['25', '30', '50']},
                   index=[0, 1, 2])
```


```python
df2 = pd.DataFrame({'name': ['Adam', 'Smith' ],
                    'Age': ['26', '11']},
                   index=[3, 4])
```

Finally, you concatenate the two DataFrame


```python
df_concat = pd.concat([df1,df2])
df_concat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smith</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paul</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adam</td>
      <td>26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Smith</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



### Drop_duplicates
If a dataset can contain duplicates information use, `drop_duplicates` is an easy to exclude duplicate rows. You can see that `df_concat` has a duplicate observation, Smith appears twice in the column name.


```python
df_concat.drop_duplicates('name')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smith</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paul</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adam</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>



### Sort values
You can sort value with `sort_values`


```python
df_concat.sort_values('Age')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Smith</td>
      <td>11</td>
    </tr>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adam</td>
      <td>26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smith</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paul</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



### Rename: change of index

You can use `rename` to rename a column in Pandas. The first value is the
current column name and the second value is the new column name.


```python
df_concat.rename(columns={"name": "Surname", "Age": "Age_ppl"})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Surname</th>
      <th>Age_ppl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smith</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paul</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adam</td>
      <td>26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Smith</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



### Import CSV

During the TensorFlow tutorial, you will use the adult dataset. It is
often used with classification task. It is available in this URL
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
The data is stored in a CSV format. This dataset includes eights
categorical variables:

This dataset includes eights categorical variables:

-   workclass

-   education

-   marital

-   occupation

-   relationship

-   race

-   sex

-   native_country

moreover, six continuous variables:

-   age

-   fnlwgt

-   education_num

-   capital_gain

-   capital_loss

hours_week

To import a CSV dataset, you can use the object pd.read_csv(). The
basic argument inside is:

-   `filepath_or_buffer`: Path or URL with the data

-   `sep=', '`: Define the delimiter to use

-   `names=None`: Name the columns. If the dataset has ten columns,
    you need to pass ten names

-   `index_col=None`: If yes, the first column is used as a row index

-   `skipinitialspace=False`: Skip spaces after delimiter.

For more information about read*csv(),* please check the official
[documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)


```python
## Import csv
import pandas as pd
## Define path data
COLUMNS = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital',
               'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
               'hours_week', 'native_country', 'label']
PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df_train = pd.read_csv(PATH,
                           skipinitialspace=True,
                           names = COLUMNS,
                           index_col=False)

df_train.shape
```




    (32561, 15)



### Groupby

An easy way to see the data is to use the `groupby` method. This method
can help you to summarize the data by group. Below is a list of methods
available with `groupby`:

-   count: `count`

-   min: `min`

-   max: `max`

-   mean: `mean`

-   median: `median`

-   standard deviation: `sdt`

-   etc

Inside
`groupby(), ``you can ``use the column you want to apply the method.`

Let's have a look at a single grouping with the adult dataset. You will
get the `mean` of all the continuous variables by type of revenue, i.e.,
above 50k or below 50k


```python
df_train.groupby(['label']).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education_num</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_week</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;=50K</th>
      <td>36.783738</td>
      <td>190340.86517</td>
      <td>9.595065</td>
      <td>148.752468</td>
      <td>53.142921</td>
      <td>38.840210</td>
    </tr>
    <tr>
      <th>&gt;50K</th>
      <td>44.249841</td>
      <td>188005.00000</td>
      <td>11.611657</td>
      <td>4006.142456</td>
      <td>195.001530</td>
      <td>45.473026</td>
    </tr>
  </tbody>
</table>
</div>



You can get the minimum of age by type of household


```python
df_train.groupby(['label'])['age'].min()
```




    label
    <=50K    17
    >50K     19
    Name: age, dtype: int64



You can also group by multiple columns. For instance, you can get the
maximum capital gain according to the household type and marital status.


```python
df_train.groupby(['label', 'marital'])['capital_gain'].max()
```




    label  marital              
    <=50K  Divorced                 34095
           Married-AF-spouse         2653
           Married-civ-spouse       41310
           Married-spouse-absent     6849
           Never-married            34095
           Separated                 7443
           Widowed                   6849
    >50K   Divorced                 99999
           Married-AF-spouse         7298
           Married-civ-spouse       99999
           Married-spouse-absent    99999
           Never-married            99999
           Separated                99999
           Widowed                  99999
    Name: capital_gain, dtype: int64



You can create a plot following groupby. One way to do it is to use a
plot after the grouping.

To create a more excellent plot, you will use unstack() after mean() so
that you have the same multilevel index, or you join the values by
revenue lower than 50k and above 50k. In this case, the plot will have
two groups instead of 14 (2*7).

If you use Jupyter Notebook, make sure to add % matplotlib inline,
otherwise, no plot will be displayed


```python
% matplotlib inline

df_plot = df_train.groupby(['label',
'marital'])['capital_gain'].mean().unstack()

df_plot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>marital</th>
      <th>Divorced</th>
      <th>Married-AF-spouse</th>
      <th>Married-civ-spouse</th>
      <th>Married-spouse-absent</th>
      <th>Never-married</th>
      <th>Separated</th>
      <th>Widowed</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;=50K</th>
      <td>140.544724</td>
      <td>204.076923</td>
      <td>219.249517</td>
      <td>106.559896</td>
      <td>99.052492</td>
      <td>117.190824</td>
      <td>149.811674</td>
    </tr>
    <tr>
      <th>&gt;50K</th>
      <td>5781.812095</td>
      <td>729.800000</td>
      <td>3678.163927</td>
      <td>6836.647059</td>
      <td>6137.576375</td>
      <td>6614.727273</td>
      <td>5071.117647</td>
    </tr>
  </tbody>
</table>
</div>



![](https://github.com/thomaspernet/Tensorflow/blob/master/tensorflow/13_pandas_v6_files/image002.png)


## Summary


Below is a summary of the most useful method for data science with
Pandas

| import data       | read_csv               |
|-------------------|------------------------|
| create series     | Series                 |
| Create Dataframe  | DataFrame              |
| Create date range | date_range             |
| return head       | head                   |
| return tail       | tail                   |
| Describe          | describe               |
| slice using name  | dataname['columnname'] |
| Slice using rows  | data_name[0:5]         |
