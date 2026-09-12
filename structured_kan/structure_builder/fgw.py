"""Source-initialized, categorical-feature adapter to the published FGW solver.

No phi representation or held-out equation is inspected here. The authors'
distance, transport solver, barycenter updates, outer stopping rule and restart
policy are retained. This is NOT the unmodified synthetic-initializer experiment.
"""
from __future__ import annotations
import math
import random
import time
from pathlib import Path
import numpy as np


def categorical_graphs(trees):
    from .native_trees import graph_from_tree
    vocabulary = set()
    def collect(t):
        vocabulary.add(t.name)
        for c in t.children:
            collect(c)
    for tree in trees:
        collect(tree)
    vocabulary = sorted(vocabulary)
    # Squared Euclidean distance = 1 for unequal labels, 0 for equal labels.
    basis = np.eye(len(vocabulary), dtype=np.float64) / np.sqrt(2.0)
    encoding = dict(zip(vocabulary, basis))
    graphs = np.asarray([graph_from_tree(t, encoding) for t in trees], dtype=object)
    return graphs, vocabulary


def make_model(clustering, out, k, numerical=None):
    from .io import write
    solver = numerical or clustering

    class SourceInitializedFGW(clustering.FusedGromovWassersteinGraphKMeans):
        def __init__(self):
            super().__init__(N=None, n_clusters=k, random_state=421, verbose=0)
            self.trace = []
            self.attempt = 0
            self.barycenter_updates = 0
            self.started = time.monotonic()

        def record(self, event, **data):
            item = dict(event=event, attempt=self.attempt,
                        elapsed_seconds=time.monotonic()-self.started, **data)
            self.trace.append(item)
            write(out/'native_progress.json', dict(latest=item,
                  barycenter_updates=self.barycenter_updates, trace=self.trace))

        def _assign_fgw(self, X, structural_information, update_class_attributes=True):
            try:
                if numerical is None:
                    result = super()._assign_fgw(X, structural_information, update_class_attributes)
                else:
                    distances = solver.cdist_fgw(
                        [x.values() for x in X], structural_information,
                        self._cluster_centers_features, self._cluster_centers_structure,
                        self.alpha_fgw)
                    result = distances.argmin(axis=1)
                    if update_class_attributes:
                        self.labels_ = result
                        clustering._check_no_empty_cluster(result, k)
                        self.inertia_ = self.compute_inertia(distances, result)
            except clustering.EmptyClusterError:
                self.record('empty_cluster_restart', occupied=len(set(self.labels_.tolist())))
                raise
            if update_class_attributes:
                assert np.isfinite(self.inertia_), 'Nonfinite FGW objective'
                self.record('assignment', inertia=float(self.inertia_),
                            cluster_sizes=np.bincount(self.labels_, minlength=k).tolist())
            return result

        def _update_centroids(self, X, it):
            # Use the seed graph's order throughout this cluster's fit. Unlike
            # native N=None, this never pairs a changed N with an old-size init.
            # All numerical barycenter operations below are the native routine.
            for j in range(k):
                group = X[self.labels_ == j]
                n = len(self._cluster_centers_structure[j])
                weights = [np.ones(len(x.nodes()))/len(x.nodes()) for x in group]
                features, structure, log = solver.fgw_barycenters(
                    N=n, Ys=[x.values() for x in group], Cs=[x.C for x in group],
                    ps=weights, lambdas=np.ones(len(group))/len(group),
                    alpha=self.alpha_fgw, max_iter=self.max_iter_barycenter,
                    fixed_structure=False, fixed_features=False,
                    init_X=self._cluster_centers_features[j],
                    init_C=self._cluster_centers_structure[j], verbose=False)
                assert features.shape == (n, X[0].values().shape[1])
                assert structure.shape == (n,n)
                assert np.isfinite(features).all() and np.isfinite(structure).all()
                assert np.max(np.abs(structure-structure.T)) < 1e-8
                marginal_error = max(max(float(np.max(np.abs(t.sum(axis=1)-1/n))),
                                         float(np.max(np.abs(t.sum(axis=0)-p))))
                                     for t,p in zip(log['T'], weights))
                assert marginal_error < 1e-7, marginal_error
                self._cluster_centers_features[j] = features
                self._cluster_centers_structure[j] = structure
                self.all_cluster_centers_features[(it,j)] = features
                self.all_cluster_centers_structure[(it,j)] = structure
                self.barycenter_updates += 1
                self.record('barycenter', iteration=it, cluster=j,
                            barycenter_iterations=len(log['err_structure']),
                            marginal_error=marginal_error, order=n,
                            final_feature_change=float(log['err_feature'][-1]),
                            final_structure_change=float(log['err_structure'][-1]))

        def _fit_one_init_fgw(self, X, structural_information, rs, y=None):
            self.attempt += 1
            indices = rs.choice(len(X), size=k, replace=False)
            self._cluster_centers_features = [X[i].values().copy() for i in indices]
            self._cluster_centers_structure = [X[i].C.copy() for i in indices]
            self.record('source_initialization', source_indices=indices.tolist(),
                        orders=[len(x) for x in self._cluster_centers_structure])
            old_inertia = np.inf
            for it in range(self.max_iter):
                self._assign_fgw(X, structural_information)
                self._update_centroids(X, it+1)
                if abs(old_inertia-self.inertia_) < self.tol:
                    self.attempted_iter = it+1
                    self.record('outer_stopped', reason='native_inertia_tolerance', iterations=it+1)
                    break
                old_inertia = self.inertia_
            else:
                self.attempted_iter = self.max_iter+1
                self.record('outer_stopped', reason='native_max_iter', iterations=self.max_iter)
            return self

    return SourceInitializedFGW()


def native_fgw_source(trees, k, vendor, out, numerical=None):
    from .vendor_api import load_fgw
    from .io import write, sha
    _, clustering = load_fgw(Path(vendor))
    graphs, vocabulary = categorical_graphs(trees)
    np.random.seed(421)
    random.seed(421)
    solver = numerical or clustering
    model = make_model(clustering, out, k, numerical=numerical)
    try:
        model.fit(graphs)
        if model.X_fit_ is None:
            raise RuntimeError('Source-initialized FGW exhausted native restart budget')
        assert model.barycenter_updates >= k
        assert len(set(model.labels_.tolist())) == k
        groups = [np.flatnonzero(model.labels_ == j).tolist() for j in range(k)]
        reps, scores = [], []
        for j, group in enumerate(groups):
            d = solver.cdist_fgw([graphs[i].values() for i in group], [graphs[i].C for i in group],
                    [model._cluster_centers_features[j]], [model._cluster_centers_structure[j]], model.alpha_fgw).ravel()
            assert np.isfinite(d).all()
            best = min(range(len(group)), key=lambda p:(float(d[p]), trees[group[p]].signature(), group[p]))
            reps.append(group[best])
            scores.append(float(d[best]))
        return groups, reps, dict(representative_distances=scores,
               adapter='FGW + categorical features + source initialization/fixed seed orders + nearest source representative',
               numerical_backend=numerical.metadata() if numerical else {'backend':'vendored published FGW'},
               barycenter_updates=model.barycenter_updates, outer_iterations=model.attempted_iter,
               attempts=model.attempt,
               final_assignment_inertia=next(x['inertia'] for x in reversed(model.trace) if x['event']=='assignment'))
    finally:
        write(out/'native.json', dict(
            protocol='categorical-source-initialized-fgw-pot-v1' if numerical else 'categorical-source-initialized-fgw-v1',
            numerical_backend=numerical.metadata() if numerical else {'backend':'vendored published FGW'},
            features=[np.asarray(x).tolist() for x in (model._cluster_centers_features or [])],
            structure=[np.asarray(x).tolist() for x in (model._cluster_centers_structure or [])],
            labels=None if model.labels_ is None else model.labels_.tolist(),
            inertia=float(model.inertia_) if math.isfinite(model.inertia_) else None,
            vocabulary=vocabulary, feature_encoding='one-hot/sqrt(2); unequal-label squared distance one',
            centroid_order_policy='source seed graph order fixed per cluster throughout each attempt',
            native_max_iter=model.max_iter, native_barycenter_max_iter=model.max_iter_barycenter,
            native_max_attempts=model.max_attempts, alpha=model.alpha_fgw,
            barycenter_updates=model.barycenter_updates, trace=model.trace,
            solver_sha256=sha(Path(vendor)/'lib'/'FGW.py'),
            clustering_sha256=sha(Path(vendor)/'lib'/'fgwclustering.py')))
