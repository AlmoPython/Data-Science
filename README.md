@[TOC](Machine Learning #1--Data Preprocessing)

# **Basic structure of data preprocessing**
When we are using machine learning to solve problems, either in a working environment or just practicing, we don't just get the cleaned and engineered data from someone else, we have to handle the raw data ourselves before putting it into the model. 
This can be time consuming but it is really important to do it and to do it correct. Because data preprocessing is the key to a successful predictive model, without proper data preprocessing, no matter how good is your model, how much you have done on parameter tuning, model evaluation, you will not get the good result you want. 
Here is the basic structure of Data Preprocessing:
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019110802314828.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUyNTA4Ng==,size_16,color_FFFFFF,t_70)
## Data Transformation
Now let's talk about the first processing technique: data transformation. 	
It can be categorized into three segments: data discretization, data generalization and data standardization.  	
#### *Data discretization.*
Data discretization assumes that the continuous features in different intervals have different contributions to the results. Different interval's features should be assigned a different weight and then each interval becomes a new feature.  	
The reason that we want to use data discretization is the following:
		1. We know that some of the models cannot use continuous variables as inputs, like Decision Tree and Naive Bayes. So we need to discrete the variables and then put them into the models. 
		2. Simplify the model: changed each new interval into a new feature will greatly decreased the numbers of features and will make the model less complicated. 
		3. data discretization can decrease the effect of outliers to the result.
There are three main methods to conduct data discretization: 
		1. Equal width binning --  divide the data in equal widths, for example: age is a continuous variables, from 1 to 100 years old. We divide it into 5 groups, each group has a 20 years gap, 1-20 years old, 21- 40 years old, etc. This is a classic example of equal width binning. 
		'width=(max−min)/N '   we only consider width here, so the number of  variables in each bin may be different. 
		2. Equal frequency binning -- groups are designed in a way such that the same or nearly the same number of continuous data points are allocated to each discrete data group. 
		Both equal width and equal frequency binning are for ***unsupervised learning*** 
		To optimize the binning: we need to introduce a standard to measure what kind of width or frequency is the best option. Normally we use: Information Value(IV), entropy and mutual information.  
For supervised learning, we use 1R and Chi-squared base method. 

#### *Data Generalization*
Data Generalization is the process of creating successive layers of summary data in an evaluational database. It is a process of zooming out to get a broader view of a problem, trend or situation. It is also known as rolling-up data. 
#### *Data Standardization*
Data Standardization is a data processing workflow that converts the structure of disparate datasets into a Common Data Format. As part of the Data Preparation field, Data Standardization deals with the transformation of datasets after the data is pulled from source systems and before it's loaded into target systems.

## Data Cleaning
Now let's talk about data cleaning. 
When we get raw data, there will be missing data, outliers and duplicate data, proper handling them is an important process to better train the models. 
#### *1. missing data*
When there are too many missing data in a column, we usually drop it to avoid bring noise into the set. 
	When the missing data are not a lot, like within 10%, we can use conditional mean or median of the column data to fill the blank, which is taking the mean or median of the data with the same label.
	There is another way to handle missing data, which is using interpolation method. 
	It uses two points ( $x_0，y_0）$，$（x_1，y_1）$to estimate the point in the middle of them. 
	There is another way to fill missing data, which is using algorithms like Random Forest, Linear Regression to predict the data and fill the blank. **This is what we usually use when there are not a lot of missing data.** 
#### *2. Outliers*
When doing machine learning, outliers or error data can be a serious problem, they will lead to Measurement error or exception conditions. actually, the best way is to drop the outliers before the next step analysis. Under some circumstances, outliers can provide the exception information of the whole system, so it is very important to detect outliers, it can give us more info on the related data set. 
Now let's talk about several ways to detect outliers. 
***1. simple statistics*** 
 We can use simple Pandas Describe to see the outlier:`
 pandas.describe()`
 Or we can make a regression graph to easily see the outlier. 

>![Picture from https://juejin.im/post/5b6a44f55188251aa8294b8c#heading-2](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91c2VyLWdvbGQtY2RuLnhpdHUuaW8vMjAxOC84LzgvMTY1MTcxZjViNWE4MzI4OA?x-oss-process=image/format,png)
Picture from ''https://juejin.im/post/5b6a44f55188251aa8294b8c#heading-2''

In the picture, the red arrow points to the outliers. 
 1.  ***3∂ policy***
  There is a constraint for this method, it is that the data set has to be normally distributed. 
under the 3∂ policy, when the point is outside the 3∂, then we can say it is an outlier. 
 2. ***clustering***
 We can also use clustering to model the data set, if there are data that is not belong to any cluster, then it will be the outlier. 
 There are more ways to detect outliers, the three ways above are the method that I use sometimes. 
 Now let's talk about the way to handle outliers. 
 1. Delete the record. 
 2. treat the oulier as a missing data. 
 3. use the average of the record before and behind the outlier to replace the   outlier. 
 4.  do not deal with it, perform data mining with outlier in  the set.

#### *3 Duplicates*
when you see duplicates, we can just use drop_duplicates() in python to solve it. 

```python
import pandas as pd
data.drop_duplicates()
```

## Data Reduction

> Data reduction is the transformation of numerical or alphabetical digital information derived empirically or experimentally into a corrected, ordered, and simplified form. The basic concept is the reduction of multitudinous amounts of data down to the meaningful parts.  ----from Wikipedia

Like Wiki said, data reduction helps us get the meaningful parts of the data set. It includes 3 parts, Numerosity reduction, attribute selection and dimentionality reduction. 
#### *1. Numerosity reduction.*
Numerosity reduction is the method that will replace the original  data with a smaller data set to make it better fit the model. The integrity of the data must be preserved, there can't be any missing of the information after the reduction. 
There are two ways to perform numerosity reduction: parametric and non-parametric. 
***1. parametric*** 
We assume that the data will fit for some model, so that we can only store the parameters instead of the data (x,y).
linear regression : $y = ax + b$
multi-linear regression: $y = a_1 + a_2x_2 + a_3x_3+....+a_nx_n$.
log-linear model: for discrete value. Estimate the possibilities of every point in the hyperspace.
***2. non-parametric***
not assume the existing of model. 
bar chart. clustering, random sampling, stratified sampling, data compression. 

#### *2.Attribute selection*
in machine learning, attribute selection is also known as " feature selection".  
Simply speaking, it is the technique to choose the features that we need, that have affections to the result to put into the model. This way, we get the meat, we get rid of the unnecessary features, we performed the data reduction. 
 We can use data visualization to see if the feature has connections with the result, or we can perform covariance analysis to see the covariance and correlation between the data and the result(label). 

#### *3. Dimentionality Reduction*

> In statistics, machine learning, and information theory, dimensionality reduction or dimension reduction is the process of reducing the number of random variables under consideration by obtaining a set of principal variables。 ---WIki

The mostly used method for diementionality reduction is PCA ( Primary Component Analysis), ICA(Independent Component Analysis) and IOSMAP. 
We will talk about the PCA , ICA and IOSMAP in the next article. 


