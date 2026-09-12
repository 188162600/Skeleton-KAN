"""Compatibility loaders, not replacement inference or optimization rules."""
import sys

def load_fgw(path):
    # API compatibility only, all original algorithm files stay byte-identical.
    import scipy.optimize.linesearch as old_linesearch
    from scipy.optimize._linesearch import scalar_search_armijo
    old_linesearch.scalar_search_armijo=scalar_search_armijo
    import scipy.sparse as sparse
    for cls in (sparse.spmatrix,sparse.sparray):
        if not hasattr(cls,'A'):cls.A=property(lambda self:self.toarray())
    import networkx as nx
    if not hasattr(nx.Graph,'node'):nx.Graph.node=property(lambda self:self._node)
    sys.path.insert(0,str(path/'lib'))
    import FGW,fgwclustering
    return FGW,fgwclustering


class GraphInput:
    def __init__(self,features,structure):self.features=features;self.C=structure
    def values(self):return self.features
    def nodes(self):return list(range(len(self.features)))

