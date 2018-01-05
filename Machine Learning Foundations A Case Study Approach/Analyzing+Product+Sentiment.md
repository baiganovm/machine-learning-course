
# Predicting sentiment from product reviews

```python
import graphlab
```

# Read some product review data

Loading reviews for a set of baby products. 


```python
products = graphlab.SFrame('amazon_baby.gl/')
```
# Build the word count vector for each review


```python
products['word_count'] = graphlab.text_analytics.count_words(products['review'])
```

# Assignemt 3 of Week 3


```python
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
```


```python
def calc_word_count(col, word):
    if word not in col:
        return 0
    else:
        return col[word]

for word in selected_words:
    products[word] = products['word_count'].apply(lambda x: calc_word_count(x, word))

```


```python
for word in selected_words:
    print (word + " - " + str(products[word].sum()))
```

    awesome - 2002
    great - 42420
    fantastic - 873
    amazing - 1305
    love - 40277
    horrible - 659
    bad - 3197
    terrible - 673
    awful - 345
    wow - 131
    hate - 1057


## Task 1 result is great = 42420


```python
train_data,test_data = products.random_split(.8, seed=0)
```


```python
selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=selected_words,
                                                     validation_set=test_data)
```


<pre>Logistic regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 133448</pre>



<pre>Number of classes           : 2</pre>



<pre>Number of feature columns   : 11</pre>



<pre>Number of unpacked features : 11</pre>



<pre>Number of coefficients    : 12</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+-------------------+---------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-accuracy | Validation-accuracy |</pre>



<pre>+-----------+----------+--------------+-------------------+---------------------+</pre>



<pre>| 1         | 2        | 0.322650     | 0.844299          | 0.842842            |</pre>



<pre>| 2         | 3        | 0.524127     | 0.844186          | 0.842842            |</pre>



<pre>| 3         | 4        | 0.727580     | 0.844276          | 0.843142            |</pre>



<pre>| 4         | 5        | 0.925202     | 0.844269          | 0.843142            |</pre>



<pre>| 5         | 6        | 1.111593     | 0.844269          | 0.843142            |</pre>



<pre>| 6         | 7        | 1.307681     | 0.844269          | 0.843142            |</pre>



<pre>+-----------+----------+--------------+-------------------+---------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
selected_words_model['coefficients'].print_rows(num_rows=12)
```

    +-------------+-------+-------+------------------+------------------+
    |     name    | index | class |      value       |      stderr      |
    +-------------+-------+-------+------------------+------------------+
    | (intercept) |  None |   1   |  1.36728315229   | 0.00861805467824 |
    |   awesome   |  None |   1   |  1.05800888878   |  0.110865296265  |
    |    great    |  None |   1   |  0.883937894898  | 0.0217379527921  |
    |  fantastic  |  None |   1   |  0.891303090304  |  0.154532343591  |
    |   amazing   |  None |   1   |  0.892802422508  |  0.127989503231  |
    |     love    |  None |   1   |  1.39989834302   | 0.0287147460124  |
    |   horrible  |  None |   1   |  -1.99651800559  | 0.0973584169028  |
    |     bad     |  None |   1   | -0.985827369929  | 0.0433603009142  |
    |   terrible  |  None |   1   |  -2.09049998487  | 0.0967241912229  |
    |    awful    |  None |   1   |  -1.76469955631  |  0.134679803365  |
    |     wow     |  None |   1   | -0.0541450123333 |  0.275616449416  |
    |     hate    |  None |   1   |  -1.40916406276  | 0.0771983993506  |
    +-------------+-------+-------+------------------+------------------+
    [12 rows x 5 columns]
    



```python
selected_words_model['coefficients'].sort('value').print_rows(num_rows=12)
```

    +-------------+-------+-------+------------------+------------------+
    |     name    | index | class |      value       |      stderr      |
    +-------------+-------+-------+------------------+------------------+
    |   terrible  |  None |   1   |  -2.09049998487  | 0.0967241912229  |
    |   horrible  |  None |   1   |  -1.99651800559  | 0.0973584169028  |
    |    awful    |  None |   1   |  -1.76469955631  |  0.134679803365  |
    |     hate    |  None |   1   |  -1.40916406276  | 0.0771983993506  |
    |     bad     |  None |   1   | -0.985827369929  | 0.0433603009142  |
    |     wow     |  None |   1   | -0.0541450123333 |  0.275616449416  |
    |    great    |  None |   1   |  0.883937894898  | 0.0217379527921  |
    |  fantastic  |  None |   1   |  0.891303090304  |  0.154532343591  |
    |   amazing   |  None |   1   |  0.892802422508  |  0.127989503231  |
    |   awesome   |  None |   1   |  1.05800888878   |  0.110865296265  |
    | (intercept) |  None |   1   |  1.36728315229   | 0.00861805467824 |
    |     love    |  None |   1   |  1.39989834302   | 0.0287147460124  |
    +-------------+-------+-------+------------------+------------------+
    [12 rows x 5 columns]
    


## Task 2 result is the most positive rate is awesome = 1.0580 and the most negative is terrible = - 2.0904


```python
sentiment_model.evaluate(test_data)

```




    {'accuracy': 0.916256305548883,
     'auc': 0.9446492867438502,
     'confusion_matrix': Columns:
     	target_label	int
     	predicted_label	int
     	count	int
     
     Rows: 4
     
     Data:
     +--------------+-----------------+-------+
     | target_label | predicted_label | count |
     +--------------+-----------------+-------+
     |      0       |        1        |  1328 |
     |      0       |        0        |  4000 |
     |      1       |        1        | 26515 |
     |      1       |        0        |  1461 |
     +--------------+-----------------+-------+
     [4 rows x 3 columns],
     'f1_score': 0.9500349343413533,
     'log_loss': 0.26106698432422165,
     'precision': 0.9523039902309378,
     'recall': 0.9477766657134686,
     'roc_curve': Columns:
     	threshold	float
     	fpr	float
     	tpr	float
     	p	int
     	n	int
     
     Rows: 100001
     
     Data:
     +-----------+----------------+----------------+-------+------+
     | threshold |      fpr       |      tpr       |   p   |  n   |
     +-----------+----------------+----------------+-------+------+
     |    0.0    |      1.0       |      1.0       | 27976 | 5328 |
     |   1e-05   | 0.909346846847 | 0.998856162425 | 27976 | 5328 |
     |   2e-05   | 0.896021021021 | 0.998748927652 | 27976 | 5328 |
     |   3e-05   | 0.886448948949 | 0.998462968259 | 27976 | 5328 |
     |   4e-05   | 0.879692192192 | 0.998284243637 | 27976 | 5328 |
     |   5e-05   | 0.875187687688 | 0.998212753789 | 27976 | 5328 |
     |   6e-05   | 0.872184684685 | 0.998177008865 | 27976 | 5328 |
     |   7e-05   | 0.868618618619 | 0.998034029168 | 27976 | 5328 |
     |   8e-05   | 0.864677177177 | 0.997998284244 | 27976 | 5328 |
     |   9e-05   | 0.860735735736 | 0.997962539319 | 27976 | 5328 |
     +-----------+----------------+----------------+-------+------+
     [100001 rows x 5 columns]
     Note: Only the head of the SFrame is printed.
     You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.}




```python
selected_words_model.evaluate(test_data)
```




    {'accuracy': 0.8431419649291376,
     'auc': 0.6648096413721418,
     'confusion_matrix': Columns:
     	target_label	int
     	predicted_label	int
     	count	int
     
     Rows: 4
     
     Data:
     +--------------+-----------------+-------+
     | target_label | predicted_label | count |
     +--------------+-----------------+-------+
     |      0       |        0        |  234  |
     |      0       |        1        |  5094 |
     |      1       |        1        | 27846 |
     |      1       |        0        |  130  |
     +--------------+-----------------+-------+
     [4 rows x 3 columns],
     'f1_score': 0.914242563530107,
     'log_loss': 0.4054747110366022,
     'precision': 0.8453551912568306,
     'recall': 0.9953531598513011,
     'roc_curve': Columns:
     	threshold	float
     	fpr	float
     	tpr	float
     	p	int
     	n	int
     
     Rows: 100001
     
     Data:
     +-----------+-----+-----+-------+------+
     | threshold | fpr | tpr |   p   |  n   |
     +-----------+-----+-----+-------+------+
     |    0.0    | 1.0 | 1.0 | 27976 | 5328 |
     |   1e-05   | 1.0 | 1.0 | 27976 | 5328 |
     |   2e-05   | 1.0 | 1.0 | 27976 | 5328 |
     |   3e-05   | 1.0 | 1.0 | 27976 | 5328 |
     |   4e-05   | 1.0 | 1.0 | 27976 | 5328 |
     |   5e-05   | 1.0 | 1.0 | 27976 | 5328 |
     |   6e-05   | 1.0 | 1.0 | 27976 | 5328 |
     |   7e-05   | 1.0 | 1.0 | 27976 | 5328 |
     |   8e-05   | 1.0 | 1.0 | 27976 | 5328 |
     |   9e-05   | 1.0 | 1.0 | 27976 | 5328 |
     +-----------+-----+-----+-------+------+
     [100001 rows x 5 columns]
     Note: Only the head of the SFrame is printed.
     You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.}



## Task 3 result is sentiment_model is more accuracy than selected_words_model


```python
diaper_champ_reviews = products[(products['name'] == 'Baby Trend Diaper Champ')]
```


```python
diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')
```


```python
diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)
```


```python
selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')
```




    dtype: float
    Rows: 1
    [0.796940851290673]




```python
diaper_champ_reviews[0]['review']
```




    'Baby Luke can turn a clean diaper to a dirty diaper in 3 seconds flat. The diaper champ turns the smelly diaper into "what diaper smell" in less time than that. I hesitated and wondered what I REALLY needed for the nursery. This is one of the best purchases we made. The champ, the baby bjorn, fluerville diaper bag, and graco pack and play bassinet all vie for the best baby purchase.Great product, easy to use, economical, effective, absolutly fabulous.UpdateI knew that I loved the champ, and useing the diaper genie at a friend\'s house REALLY reinforced that!! There is no comparison, the chanp is easy and smell free, the genie was difficult to use one handed (which is absolutly vital if you have a little one on a changing pad) and there was a deffinite odor eminating from the genieplus we found that the quick tie garbage bags where the ties are integrated into the bag work really well because there isn\'t any added bulk around the sealing edge of the champ.'


