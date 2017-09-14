## ed_viz - Tensorflow/Edward Graph Visualize 

A small python script to visualize tensorflow graph especially for edward (edwardlib.org).
Requires the Graphviz software (Graphviz.org) and python graphiz library (pip install graphviz)

pip install git+https://github.com/Arvinds-ds/ed_viz.git
```
import ed_viz

ed_viz.visualize_simple()
ed_viz.visualize_full(depth=2)
```
