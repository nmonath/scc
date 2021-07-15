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

import time

import numpy as np
from absl import logging
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

logging.set_verbosity(logging.INFO)

class TreeLevel(object):

    def __init__(self, tau, dist_graph, node_counts, cluster_assignments=None, cc_connection='weak'):
        self.dist_graph = dist_graph
        self.node_counts = node_counts
        # parents[i] gives the id of the node in next round of node i
        self.parents = None
        self.num_uniq_parents = None
        self.nn_edges = None
        self.nn_edge_sims = None
        self.tau = tau
        self.cc_connection=cc_connection
        # cluster_assignments[j] gives cluster id of jth point in this round
        if cluster_assignments is None:
          self.cluster_assignments = np.arange(self.dist_graph.shape[0],dtype=np.int32)
        else:
          self.cluster_assignments = cluster_assignments

    def perform_round(self):
        row = self.dist_graph.row
        col = self.dist_graph.col
        if row.shape[0] > 1:
            oneNN_s = time.time()
            # normalize the data by counts
            data = self.dist_graph.data.copy()
            data /= (self.node_counts[row] * self.node_counts[col])
            to_use_binary = np.logical_and(data >= self.tau, row != col)
            g = csr_matrix((data[to_use_binary],
                            (row[to_use_binary],
                             col[to_use_binary])),
                           shape=self.dist_graph.shape, dtype=np.float32)

            nns = g.argmax(axis=1).A.squeeze(1)
            nns_sims = g.max(axis=1).A.squeeze(1)
            next_round_binary = coo_matrix(
                (np.ones(self.dist_graph.shape[0], dtype=np.float32)[nns_sims != 0],
                 (np.arange(self.dist_graph.shape[0], dtype=np.int32)[nns_sims != 0],
                  nns[nns_sims != 0])),
                shape=self.dist_graph.shape)
            next_round_binary.eliminate_zeros()
            oneNN_e = time.time()
            logging.debug('Nearest Neighbor: Done. nodes %s, edges %s, time %s', self.dist_graph.shape[0], row.shape[0], oneNN_e-oneNN_s)
            CC_s = time.time()
            num_uniq_parents, parent_map = connected_components(next_round_binary,
                                                                directed=False, connection='weak')
            CC_e = time.time()
            logging.debug('Connected Components: Done. nodes %s, edges %s, time %s', next_round_binary.shape[0], next_round_binary.nnz,
                         CC_e - CC_s)
            self.parents = parent_map
            self.num_uniq_parents = num_uniq_parents
        else:
            self.num_uniq_parents = 1
            self.parents = np.zeros(self.dist_graph.col.shape[0], dtype=np.int32)


    def form_next_round(self, next_tau):
        if self.num_uniq_parents > 1:
            contract_s = time.time()
            next_row = self.parents[self.dist_graph.row]
            next_col = self.parents[self.dist_graph.col]
            i, j = next_row[next_row != next_col], next_col[next_row != next_col]
            next_round_dist_sum = coo_matrix((self.dist_graph.data[next_row != next_col], (i,j)),
                                             shape=(self.num_uniq_parents, self.num_uniq_parents), dtype=np.float32)
            next_round_dist_sum.sum_duplicates()


            next_counts_nodes = coo_matrix((self.node_counts, (self.parents, np.zeros_like(self.parents, dtype=np.float32))),
                                           shape=(self.num_uniq_parents, 1), dtype=np.float32)
            next_counts_nodes.sum_duplicates()
            next_counts_nodes = next_counts_nodes.todense().A[:, 0]
            next_cluster_assignments = self.parents[self.cluster_assignments]
            contract_t = time.time()
            logging.debug('Graph Contract: Done. nodes %s, edges %s, time %s', next_round_dist_sum.shape[0],
                         next_round_dist_sum.nnz,
                         contract_t - contract_s)
            return TreeLevel(next_tau, dist_graph=next_round_dist_sum,
                             node_counts=next_counts_nodes,
                             cluster_assignments=next_cluster_assignments,
                             cc_connection=self.cc_connection)
        else:
            return None

class SCC(object):
    def __init__(self, g, num_rounds, taus, cc_connection='weak'):
        self.g = g
        self.uniq_ids = np.unique(g.row)
        self.num_rounds = num_rounds
        self.taus = taus
        self.rounds = []
        self.cc_connection = cc_connection

    def assignments_by_threshold(self, threshold):
        closest_v = np.Inf
        closest = 0
        for i,r in enumerate(self.rounds):
            t = self.taus[i]
            v = np.abs(t-threshold)
            if v < closest_v:
                closest = i
                closest_v = v
        return self.rounds[closest].cluster_assignments

    def assignments_by_num_clusters(self, k):
        closest_v = np.Inf
        closest = 0
        for i,r in enumerate(self.rounds):
            v = np.abs(r.num_uniq_parents-k)
            if v < closest_v:
                closest = i
                closest_v = v
        return self.rounds[closest].cluster_assignments

    def fit(self):
        st = time.time()
        self.rounds.append(TreeLevel(tau=self.taus[0],
                                     dist_graph=self.g.copy(),
                                     node_counts=np.ones(self.g.shape[0]),
                                     cc_connection=self.cc_connection))
        for i in range(self.num_rounds):
            logging.debug('round %s', i)
            logging.debug('round %s starts with %s nodes', i, self.rounds[i].dist_graph.shape[0])
            self.rounds[i].perform_round()
            if i != self.num_rounds-1:
                nr = self.rounds[i].form_next_round(self.taus[i+1])
                if nr is not None and nr.dist_graph.nnz > 0 and nr.dist_graph.shape[0] > 1:
                    logging.debug('round %s ends with %s nodes',i,  nr.dist_graph.shape[0])
                    self.rounds.append(nr)
                else:
                    break
        ent = time.time()
        logging.debug('SCC time done in %s', ent-st)

    def write_tsv(self, outfile, labels):
        logging.debug('writing tsv tree @ %s', outfile)
        with open(outfile, 'w') as fout:
            for round_i, round_obj in tqdm(enumerate(self.rounds)):

                for j in range(round_obj.parents.shape[0]):
                    j_id = '%s_%s' % (round_i, j)

                    if round_i != len(self.rounds) - 1:
                        parent = round_obj.parents[j]
                        par_j_id = '%s_%s' % (round_i+1, parent)
                    else:
                        par_j_id = 'root'

                    lbl = 'None' if round_i > 0 else labels[j]
                    fout.write('%s\t%s\t%s\n' % (j_id, par_j_id, lbl))

            fout.write('root\tNone\tNone\n')
            logging.debug('writing fininshed!')


    def write_and_prune(self, outfile, labels):
        logging.debug('writing tsv tree @ %s', outfile)
        skip_map = dict()
        with open(outfile, 'w') as fout:
            for round_i, round_obj in tqdm(enumerate(self.rounds)):
                # check to see how many children have each parent
                # for each parent that has a single child.
                # set skip_map[parent] = skip_map[child] if child in skip_map else child
                import collections
                parent_counts = collections.defaultdict(int)
                for j in range(round_obj.parents.shape[0]):
                    if round_i != len(self.rounds) - 1:
                        parent = round_obj.parents[j]
                        par_j_id = '%s_%s' % (round_i + 1, parent)
                    else:
                        par_j_id = 'root'
                    parent_counts[par_j_id] += 1

                for j in range(round_obj.parents.shape[0]):
                    j_id = '%s_%s' % (round_i, j) if round_i > 0 else '%s' % (j)

                    if round_i != len(self.rounds) - 1:
                        parent = round_obj.parents[j]
                        par_j_id = '%s_%s' % (round_i + 1, parent)
                    else:
                        par_j_id = 'root'

                    if parent_counts[par_j_id] == 1 and round_i > 0 and par_j_id != 'root': # always write data points
                        skip_map[par_j_id] = skip_map[j_id] if j_id in skip_map else j_id
                    else:
                        lbl = 'None' if round_i > 0 else labels[j]
                        if par_j_id in skip_map:
                            par_j_id = skip_map[par_j_id]
                        if j_id in skip_map:
                            j_id = skip_map[j_id]

                        fout.write('%s\t%s\t%s\n' % (j_id, par_j_id, lbl))

            fout.write('root\tNone\tNone\n')
            logging.debug('writing fininshed!')


class Affinity(SCC):
    def __init__(self, g, num_rounds):
        super(Affinity, self).__init__(g, num_rounds, taus=-np.Inf*np.ones(num_rounds))

class RecipNN(SCC):
    def __init__(self, g, num_rounds):
        super(RecipNN, self).__init__(g, num_rounds, taus=-np.Inf*np.ones(num_rounds), cc_connection='strong')
