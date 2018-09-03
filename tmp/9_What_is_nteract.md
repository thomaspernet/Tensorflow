
# What is `nteract`


`nteract` is an interactive computing application built on top of the
Jupyter notebook. If you are familiar with Jupyter notebook, then you
will see lots of similarities with `nteract`. `nteract` is developed and
maintained by Netflix. The idea is to push the limited of Jupyter
notebook. It is oriented with a minimalist, user-centered notebook, easy
to share and publish and easy to use for a beginner. One of the first
objectives of `nteract` is to reduce the entry barrier of coding. The
ideal audience for `nteract` is data scientists, journalists, students,
and researchers.

## Install `nteract`


Install `nteract` on a local machine is not difficult. To make this
textbook consistent, we will install the library inside Tensorflow
environment.

Open the Terminal or Anaconda prompt depending if you are using Mac or
Windows. Activate the environment, and then you can use `pip` command to
install `nteract`

    pip install nteract-on-jupyter

You also need to install another library to allow the visualization
tool:

    pip install vdom==0.5

All right, you are all set. Let's use `nteract` with Jupyter notebook.

Every time you want to use `nteract`, use `jupyter nteract`

You will be redirected to your favorite browser; you should be able to
see the `nteract` interface.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image001.png)

Click on the Python logo to create a new Notebook.

Here is the `nteract` notebook.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image002.png)

You can edit, view, add cells, run cells, similar to Jupyter. Let's
write a simple code `print('hello world!')` and click on the `play`
button to run the cells.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image003.png)

Then, you can use the button below to add a new code cell. If you want
to create a text cell, click on `m+`

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image004.png)

You can create a new code that prints `Drag and drop`.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image005.png)

One nice feature of `nteract` is the drag and drop cells. You can move
the cells up or down very quickly.


![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/imagehttps://media.giphy.com/media/31ZnAfIJRbd1rsP5Zm/giphy.gif" width="400" height="400" />

You can save and close the notebook.

In the next section, we will see how to use `nteract` to create
compelling data visualization.

## A brief introduction to `nteract`


`nteract` is a different tool than Tableau, Power BI or traditional
business intelligence tools. `nteract` data explorer is not meant to
become a report or a dashboard. Its primary purpose is to get an
overview of the data quickly. When you know what graphs or chart you
want, then traditional tools are preferable. If, however, you need to
get to know the dataset, Data Explorer is a recommended tool. Data
Explorer provides an array of the way to visualize the data, such as a
bar chart, scatterplot, violin plot, heat map, etc. It said it
summarizes the data fast and with the appropriate way.

Let's see how the Data explorer works. You will use the `house` dataset
available on my GitHub. This dataset is a collection of housing price in
California during the year 1990.

This dataset gathers 11 features:

| Feature name     | type        | description                 |
|------------------|-------------|-----------------------------|
| longitude        | continuous  | Longitude                   |
| latitude         | continuous  | latitude                    |
| housingmedianage | continuous  | The median age of the house |
| total_rooms      | continuous  | Total rooms                 |
| total_bedrooms   | continuous  | Total bedroom               |
| population       | continuous  | Population                  |
| households       | continuous  | Number of households        |
| median_income    | continuous  | Median income               |
| ocean_proximity  | categorical | Close to the ocean          |
| medianhousevalue | continuous  | Median house value          |

The task of this dataset is to build a model to predict the housing
price in California. Each row regroups information on population, median
income, median housing price and so on. The median housing price is
computing price is computed with a typical group of 600 to 3.000 people.

Your common practice to load the data in memory is to use Pandas
library. You import the data from GitHub as `df`. After you run the
code, you should see the data

    import pandas as pd

    PATH = "https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/house_train.csv"

    df = pd.read_csv(PATH)
    df.head(2)

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image006.png)

You need to configure the notebook to allow the data explorer:



    from vdom import pre
    pd.options.display.html.table_schema = True
    pd.options.display.max_rows = None

    display(pre("Note: This data is sampled"))
    df

There is different data visualization mode in the notebook. In the image
below, each icon represents a visualization technique.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image007.png)

[source](https://blog.nteract.io/designing-the-nteract-data-explorer-f4476d53f897?gi=688ff22fcc27)

**Bar charts**

A simple bar chart is the first visualization technique. You can choose
different continuous variables on the *Metric* drop-down menu and
display it according to one or more multiple categorical variables. If
you choose a category, it will sum the value of the continuous variable
for each group.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image008.png)

You can zoom in the column to see in more detail the value of the rows.
It is an interesting technique to get detail insight from the data

**Summary chart**

The summary chart is an excellent mode to visualize the different
quartile of the data. You can detect outliers and have an idea about the
distribution of the data.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image009.png)

You can choose different plots like the violin plot (default), histogram
or heatmap

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image010.png)

**Scatterplots**

A popular tool to visualize the data is the scatterplot. You can create
a simple scatter but also add different information such as the size of
the bubble and labels to display **abnormal** points.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image011.png)

**Hexbin**

If you are interested in the density of the point, then the hexbin or
contour plot are the right tools. This plot shows the aggregated points
on hover.

![](/Users/Thomas/Dropbox/Learning/GitHub/project/Tensorflow/image/tensorflow/9_What_is_nteract_files/image012.png)

`nteract` with the Data Explorer is an excellent place to start the data
visualization. It is a good practice to learn the pattern of the data
before to go further. In this chapter, you have learned one efficient
tool before the analysis. In the next chapter, you will see a different
tool developed by Google to get fast insight from your data.
