# Sub-Cluster Component Clustering Algorithm

This is a numpy / python implementation of SCC. For relatively sparse graph inputs, it should scale
relatively easily to datasets of millions of nodes. This implementation assumes similarities 
are given. 

There is an example use in [demo.py](examples/demo.py). This demo shows:

```Python
upper = 1.0
lower = 0.1
num_rounds = 50
X = np.random.randn(100,5)
graph = graph_from_vectors(X, k=25, batch_size=5000)
taus = np.geomspace(start=upper, stop=lower, num=num_rounds)

scc = SCC(graph, num_rounds, taus)
scc.fit()

# How to inspect this? 
# this gives the things stored in the 3rd round of the alg.  (0 based)
scc.rounds[3].__dict__

# the cluster assignment of the 18th point of the dataset. (0 based)
scc.rounds[3].cluster_assignments[18]

# the id of the parent in the next round of node 2 (0 based)
scc.rounds[3].parents[2]
```


Citation: 

```
@article{scc2020arxiv,
  author    = {Nicholas Monath and
               Avinava Dubey and
               Guru Guruganesh and
               Manzil Zaheer and
               Amr Ahmed and
               Andrew McCallum and
               G{\"{o}}khan Mergen and
               Marc Najork and
               Mert Terzihan and
               Bryon Tjanaka and
               Yuan Wang and
               Yuchen Wu},
  title     = {Scalable Bottom-Up Hierarchical Clustering},
  journal   = {arXiv preprint, 2010.11821},
  year      = {2020}
}
```

