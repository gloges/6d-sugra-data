# 6d-sugra-data

Companion data to [gloges/6d-sugra-classify](https://github.com/gloges/6d-sugra-classify/), containing the building blocks to build anomaly-free 6d supergravity models. See also

 - Preprint: [[2311.00868](https://arxiv.org/abs/2311.00868)]
 - Article: [10.1007/JHEP02(2024)095](https://link.springer.com/article/10.1007/JHEP02(2024)095)

for more information.

Irreps are organized by group type (A/B/C/D/E) and rank. Each line contains the irrep name, dimension, indices A,B,C, whether the it is quaternionic and its highest weight vector.

Vertices are again organized by group and each line contains a unique identifier, values for $\Delta = H_\text{ch} - V$, $b_i\cdot b_i$, $b_0\cdot b_i$ and irrep multiplicities.

Cliques are organized by numbers of type A and B vertices (see papers linked above). Their data include a unique identifier, constituent vertices, values for $\Delta$ and $\Delta+28n_-^g$, minimal allowed value for $T$ and a complete list of hypermultiplet representations.

Some functions for loading in and searching the data are provided in [src/helper.py](https://github.com/gloges/6d-sugra-data/blob/1a1a4cd1637eec5b153b3067e5a18aaa7741b7f7/src/helper.py) and exemplified in the iPython notebook [example-T=0.ipynb](https://github.com/gloges/6d-sugra-data/blob/1a1a4cd1637eec5b153b3067e5a18aaa7741b7f7/example-T%3D0.ipynb).


***

Related work: in a follow-up work we more thoroughly analyzed models with $T\leq 1$. See [[2404.08845](https://arxiv.org/abs/2404.08845)] and accompanying data [gloges/6d-sugra-data-T01](https://github.com/gloges/6d-sugra-data-T01).
