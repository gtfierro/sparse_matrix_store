from rdflib import URIRef, BNode, Literal
from rdflib.store import Store
from rdflib.term import Node
from rdflib.plugin import register
import scipy.sparse as sp
from scipy.sparse import dok_matrix
from rdflib.graph import Graph


class SparseMatrixStore(Store):
    """
    An RDF store implemented using a sparse matrix backend with preallocated capacity.

    Attributes:
        node2index (dict): Maps a node (subject/object) to its unique integer index.
        index2node (dict): Reverse mapping of indices back to nodes.
        matrix_size (int): The number of nodes currently in the store.
        matrix_capacity (int): The allocated capacity for matrix dimensions.
        pred_matrices (dict): Maps predicates to their corresponding sparse dok_matrix.
                              Each matrix has shape (matrix_capacity, matrix_capacity).
    """

    def __init__(self, configuration=None, identifier=None, prealloc_capacity=100000):
        super().__init__(configuration)
        self.identifier = identifier
        self.node2index = {}
        self.index2node = {}
        self.matrix_size = 0
        self.matrix_capacity = prealloc_capacity
        self.pred_matrices = {}

    def open(self, configuration, create=False):
        """Open store (no-op for this implementation)"""
        return Store.VALID_STORE

    def close(self, commit_pending_transaction=False):
        """Close store (no-op for this implementation)"""
        pass

    def destroy(self):
        """Destroy store (clear all data)"""
        self.node2index.clear()
        self.index2node.clear()
        self.pred_matrices.clear()
        self.matrix_size = 0

    def _expand_matrices_to(self, new_capacity):
        for pred, mat in self.pred_matrices.items():
            new_mat = dok_matrix((new_capacity, new_capacity), dtype=bool)
            for (i, j), val in mat.items():
                new_mat[i, j] = val
            self.pred_matrices[pred] = new_mat
        self.matrix_capacity = new_capacity

    def _get_node_index(self, node: Node) -> int:
        if node not in self.node2index:
            index = self.matrix_size
            self.node2index[node] = index
            self.index2node[index] = node
            self.matrix_size += 1
            if self.matrix_size >= self.matrix_capacity:
                self._expand_matrices_to(self.matrix_capacity * 2)
        return self.node2index[node]

    def add(self, triple, context, quoted=False):
        s, p, o = triple
        s_index = self._get_node_index(s)
        o_index = self._get_node_index(o)

        if p not in self.pred_matrices:
            self.pred_matrices[p] = dok_matrix(
                (self.matrix_capacity, self.matrix_capacity), dtype=bool
            )
        else:
            if self.pred_matrices[p].shape[0] < self.matrix_capacity:
                self._expand_matrices_to(self.matrix_capacity)

        self.pred_matrices[p][s_index, o_index] = True

    def remove(self, triple, context=None):
        s, p, o = triple
        if p not in self.pred_matrices:
            return
        try:
            s_index = self.node2index[s]
            o_index = self.node2index[o]
        except KeyError:
            return

        mat = self.pred_matrices[p]
        if (s_index, o_index) in mat:
            del mat[s_index, o_index]

    def triples(self, triple_pattern, context=None):
        """Retrieve triples matching the triple pattern."""
        s, p, o = triple_pattern

        if p is not None and p in self.pred_matrices:
            if s is not None:
                if isinstance(s, Node) and s in self.node2index:
                    s_index = self.node2index[s]
                else:
                    return
            else:
                s_index = slice(None)

            if o is not None:
                if isinstance(o, Node) and o in self.node2index:
                    o_index = self.node2index[o]
                else:
                    return
            else:
                o_index = slice(None)

            for i, j in self.pred_matrices[p].keys():
                if (s_index == slice(None) or i == s_index) and (
                    o_index == slice(None) or j == o_index
                ):
                    yield (self.index2node[i], p, self.index2node[j]), None
        elif p is None:
            for predicate in self.pred_matrices:
                yield from self.triples((s, predicate, o), context)

    def __len__(self, context=None):
        return sum(len(mat) for mat in self.pred_matrices.values())

    def __contains__(self, triple):
        s, p, o = triple
        if (
            p not in self.pred_matrices
            or s not in self.node2index
            or o not in self.node2index
        ):
            return False
        s_index = self.node2index[s]
        o_index = self.node2index[o]
        return (s_index, o_index) in self.pred_matrices[p]


# Register the SparseMatrixStore as an rdflib Store plugin.
register("sparse_matrix", Store, __name__, "SparseMatrixStore")

# Example usage
if __name__ == "__main__":
    store = SparseMatrixStore()
    graph = Graph(store=store)

    s1 = URIRef("http://example.org/subject1")
    p_knows = URIRef("http://xmlns.com/foaf/0.1/knows")
    o1 = URIRef("http://example.org/object1")

    graph.add((s1, p_knows, o1))

    print("Graph contains:")
    for triple in graph:
        print(triple)

    print("Triple count:", len(graph))
