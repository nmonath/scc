"""
 Copyright (c) 2021 The authors of SCC All rights reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
     
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os

import numpy as np
from absl import app
from absl import logging
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm
from scc import SCC

import numpy.random
import time

logging.set_verbosity(logging.INFO)

def sim_fn(XA,XB):
  return XA @ XB.T

def batched_knn(XA, XB, K, batch_size=1000, offset=0):
    K = np.minimum(K, XB.shape[0])
    res_i = np.zeros((XA.shape[0], K), dtype=np.int32)
    res = np.zeros((XA.shape[0], K), dtype=np.int32)
    resd = np.zeros((XA.shape[0], K), dtype=np.float32)
    for i in tqdm([x for x in range(0, XA.shape[0], batch_size)]):
        istart = i
        iend = min(XA.shape[0], i + batch_size)
        r = np.zeros((iend-istart, XB.shape[0]), dtype=np.float32)
        for j in range(0, XB.shape[0], batch_size):
            jstart = j
            jend = min(XB.shape[0], j + batch_size)
            r[:, jstart:jend] = sim_fn(XA[istart:iend], XB[jstart:jend])
        np.put(r, np.arange(iend - istart)*r.shape[1] + np.arange(istart, iend), np.inf)
        res[istart:iend, :] = np.argpartition(r, -K, axis=1)[:, -K:]
        resd[istart:iend, :] = r[np.arange(iend-istart)[:, None], res[istart:iend, :]]
        res_i[istart:iend, :] = np.repeat(np.expand_dims(np.arange(istart, iend), 1), K, axis=1) + offset

    row = res_i.flatten()
    col = res.flatten()
    d = resd.flatten()
    c = coo_matrix((d[row!=col], (row[row!=col], col[row!=col])), dtype=np.float32,shape=(XB.shape[0], XB.shape[0]))
    return c

def make_symmetric(coo_mat):
    lil = coo_mat.tolil()
    rows, cols = lil.nonzero()
    lil[cols, rows] = lil[rows, cols].maximum(lil[cols, rows])
    return lil.tocoo()

def graph_from_vectors(vectors, k, batch_size, random_noise=0):
    graph = batched_knn(vectors, vectors, k,offset=0, batch_size=batch_size)
    graph.data += np.random.random(graph.data.shape) * random_noise
    graph = make_symmetric(graph)
    return graph


def main(argv):
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
    print('Third round of the alg: ')
    print(scc.rounds[3].__dict__)

    # the cluster assignment of the 18th point of the dataset. (0 based)
    print('cluster assignment of the 18th point of the dataset')
    print('scc.rounds[3].cluster_assignments[18]')
    print(scc.rounds[3].cluster_assignments[18])

    # the id of the parent in the next round of node 0 (0 based)
    print('the id of the parent in the next round of node 0 (0 based)')
    print('scc.rounds[3].parents[0]')
    print(scc.rounds[3].parents[0])

if __name__ == "__main__":
   app.run(main)
