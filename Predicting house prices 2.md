
# Fire up graphlab create


```python
import graphlab
```


# Load some house sales data



```python
sales = graphlab.SFrame('home_data.gl/')
```



```python
sales

```



```python

```

# Exploring the data for housing sales


```python
graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living", y="price")
```



# Create a simple regression model of sqft_living to price


```python
train_data, test_data = sales.random_split(.8,seed=0)
```


# Assignment 2 of Week 2

## Task 1


```python
zip = graphlab.SFrame(sales[sales['zipcode']=='98039'])
```


```python
zip['price'].mean()
```




    2160606.5999999996



## Task 2


```python
range = graphlab.SFrame(sales[(sales['sqft_living'] > 2000) & (sales['sqft_living'] < 4000)])
```


```python
print float(range.num_rows()) / sales.num_rows()
```

    0.421551843798


## Task 3


```python
my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
```


```python
my_features_model = graphlab.linear_regression.create(train_data, target='price', features=my_features)

```

    PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.
              You can set ``validation_set=None`` to disable validation tracking.
    



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 16478</pre>



<pre>Number of features          : 6</pre>



<pre>Number of unpacked features : 6</pre>



<pre>Number of coefficients    : 115</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Validation-max_error | Training-rmse | Validation-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+</pre>



<pre>| 1         | 2        | 0.150501     | 3765734.606913     | 2817292.272171       | 182221.395850 | 179167.367430   |</pre>



<pre>+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
fea_res = my_features_model.evaluate(test_data)
```


```python
print fea_res
```

    {'max_error': 3503614.5340737207, 'rmse': 179749.10931802166}



```python
advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]
```


```python
advanced_features_model = graphlab.linear_regression.create(train_data, target='price', features=advanced_features)
```

    PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.
              You can set ``validation_set=None`` to disable validation tracking.
    



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 16496</pre>



<pre>Number of features          : 18</pre>



<pre>Number of unpacked features : 18</pre>



<pre>Number of coefficients    : 126</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Validation-max_error | Training-rmse | Validation-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+</pre>



<pre>| 1         | 2        | 0.100567     | 3459405.816107     | 4678544.500785       | 152800.923776 | 211595.151659   |</pre>



<pre>+-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
adv_res = advanced_features_model.evaluate(test_data)
```


```python
print adv_res
```

    {'max_error': 3587150.2232304565, 'rmse': 157314.7585669548}



```python
fea_res['rmse'] - adv_res['rmse']
```




    22434.350751066842




```python

```
