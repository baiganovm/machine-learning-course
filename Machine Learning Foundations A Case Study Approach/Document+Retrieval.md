
# Document retrieval from wikipedia data


```python
import graphlab
```
# Load some text data - from wikipedia, pages on people


```python
people = graphlab.SFrame('people_wiki.gl/')
```

# Compute TF-IDF for the corpus 



```python
people['word_count'] = graphlab.text_analytics.count_words(people['text'])
```


```python
tfidf = graphlab.text_analytics.tf_idf(people['word_count'])

# Earlier versions of GraphLab Create returned an SFrame rather than a single SArray
# This notebook was created using Graphlab Create version 1.7.1
if graphlab.version <= '1.6.1':
    tfidf = tfidf['docs']

tfidf
```

```python
people['tfidf'] = tfidf
```


# Assignment of Week 4


```python
elton = people[people['name'] == 'Elton John']
```



```python
elton_word_count_table = elton[['word_count']].stack('word_count', new_column_name = ['word','count'])
```


```python
elton_word_count_table.sort('count',ascending=False)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">27</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">15</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">has</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">john</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">on</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">award</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[255 rows x 2 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
elton[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tfidf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">furnish</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18.38947184</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">elton</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17.48232027</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">billboard</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17.3036809575</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">john</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13.9393127924</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">songwriters</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11.250406447</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">tonightcandle</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.9864953892</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">overallelton</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.9864953892</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">19702000</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.2933482087</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">fivedecade</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.2933482087</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">aids</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.262846934</td>
    </tr>
</table>
[255 rows x 2 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



## Task 2


```python
v_beckham = people[people['name'] == 'Victoria Beckham']
```


```python
graphlab.distances.cosine(elton['tfidf'][0],v_beckham['tfidf'][0])
```




    0.9567006376655429




```python
poul = people[people['name'] == 'Paul McCartney']
```


```python
graphlab.distances.cosine(elton['tfidf'][0],poul['tfidf'][0])
```




    0.8250310029221779



## Task 3


```python
knn_model_n = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name',distance='cosine')
```


<pre>Starting brute force nearest neighbors model training.</pre>



```python
knn_model_n.query(elton)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 16.914ms     |</pre>



<pre>| Done         |         | 100         | 959.578ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Elton John</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2.22044604925e-16</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Rod Stewart</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.717219667893</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">George Michael</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.747600998969</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sting (musician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.747671954431</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Phil Collins</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.75119324879</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[5 rows x 4 columns]<br/>
</div>




```python
knn_model_n.query(v_beckham)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 23.291ms     |</pre>



<pre>| Done         |         | 100         | 912.837ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Victoria Beckham</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.11022302463e-16</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">David Beckham</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.548169610263</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stephen Dow Beckham</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.784986706828</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Mel B</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.809585523409</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Caroline Rush</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.819826422919</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[5 rows x 4 columns]<br/>
</div>




```python
knn_model_nwc = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name',distance='cosine')
```


<pre>Starting brute force nearest neighbors model training.</pre>



```python
knn_model_nwc.query(elton)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 13.706ms     |</pre>



<pre>| Done         |         | 100         | 663.066ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Elton John</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.22044604925e-16</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cliff Richard</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.16142415259</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sandro Petrone</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.16822542751</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Rod Stewart</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.168327165587</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Malachi O'Doherty</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.177315545979</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[5 rows x 4 columns]<br/>
</div>




```python
knn_model_nwc.query(v_beckham)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 12.586ms     |</pre>



<pre>| Done         |         | 100         | 646.812ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Victoria Beckham</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2.22044604925e-16</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Mary Fitzgerald (artist)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.207307036115</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Adrienne Corri</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.214509782788</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Beverly Jane Fry</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.217466468741</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Raman Mundair</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.217695474992</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[5 rows x 4 columns]<br/>
</div>




```python

```


```python

```
