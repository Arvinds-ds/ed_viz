## ed_viz - Tensorflow/Edward Graph Visualize 

A small python script to visualize tensorflow graph especially for edward (edwardlib.org).
Requires the Graphviz software (Graphviz.org) and python graphiz library (pip install graphviz)

`pip install git+https://github.com/Arvinds-ds/ed_viz.git`


```
import ed_viz

from edward.models import Normal

X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D), name='w')
b = Normal(loc=tf.zeros(1), scale=tf.ones(1), name='b')
y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N)*0.1,name='y')

```

```

ed_viz.visualize_simple()

```

['simple'](img/1.png)

```
ed_viz.visualize_full(depth=1)

```
['full'](img/2.png)