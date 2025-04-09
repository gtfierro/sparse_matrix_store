"""
A custom RDF graph store that uses a sparse matrix representation,
implemented as an rdflib.Graph plugin, with preallocation of the matrix size.

Each predicate in the RDF graph is represented as a SciPy dok_matrix,
with rows corresponding to subject node indices and columns to object node indices.
An internal mapping (node2index and index2node) assigns a unique integer to each node.
Matrices are preallocated to have dimensions (prealloc_capacity x prealloc_capacity)
by default (100,000 x 100,000), and they are expanded (doubled) if more nodes are added.

The union operator efficiently computes the union of two graphs by merging
their underlying sparse matrices, reindexing nodes as necessary.
"""

from rdflib import Graph, URIRef, BNode, Literal
from rdflib.store import Store
from rdflib.term import Node
from rdflib.plugin import register
import scipy.sparse as sp
from scipy.sparse import dok_matrix

class SparseMatrixGraph(Graph):
    """
    An RDF graph implemented using a sparse matrix backend with preallocated capacity.

    Attributes:
        node2index (dict): Maps a node (subject/object) to its unique integer index.
        index2node (dict): Reverse mapping of indices back to nodes.
        matrix_size (int): The number of nodes currently in the graph.
        matrix_capacity (int): The allocated capacity for matrix dimensions.
        pred_matrices (dict): Maps predicates to their corresponding sparse dok_matrix.
                              Each matrix has shape (matrix_capacity, matrix_capacity).
    """

    def __init__(self, *args, prealloc_capacity=100000, **kwargs):
        # Initialize the base Graph.
        super().__init__(*args, **kwargs)
        # Mapping from nodes to unique integer indices.
        self.node2index = {}
        # Reverse mapping from indices to nodes.
        self.index2node = {}
        # Number of nodes added so far.
        self.matrix_size = 0
        # Preallocated capacity for the matrices.
        self.matrix_capacity = prealloc_capacity
        # Dictionary mapping each predicate to a dok_matrix (initially empty).
        self.pred_matrices = {}

    def _expand_matrices_to(self, new_capacity):
        """
        Expand all predicate matrices to have new dimensions (new_capacity x new_capacity).

        This is called when the current number of nodes reaches the allocated capacity.
        The method creates a new matrix for each predicate and copies over the existing entries.

        Args:
            new_capacity (int): The new capacity for the matrices.
        """
        for pred, mat in self.pred_matrices.items():
            new_mat = dok_matrix((new_capacity, new_capacity), dtype=bool)
            # Copy existing nonzero entries to the new matrix.
            for (i, j), val in mat.items():
                new_mat[i, j] = val
            self.pred_matrices[pred] = new_mat
        self.matrix_capacity = new_capacity

    def _get_node_index(self, node: Node) -> int:
        """
        Retrieve the index for a node; if the node doesn't exist, assign a new one.

        If adding a new node exceeds the current preallocated capacity, all predicate
        matrices are expanded (capacity is doubled).

        Args:
            node (Node): An RDFLib node (URIRef, BNode, or Literal).

        Returns:
            int: The unique index corresponding to the node.
        """
        if node not in self.node2index:
            index = self.matrix_size
            self.node2index[node] = index
            self.index2node[index] = node
            self.matrix_size += 1
            # If we've reached the preallocated capacity, expand all matrices.
            if self.matrix_size >= self.matrix_capacity:
                self._expand_matrices_to(self.matrix_capacity * 2)
        return self.node2index[node]

    def add(self, triple):
        """
        Add a triple (subject, predicate, object) to the graph.

        The subject and object nodes are assigned indices via _get_node_index.
        The predicate’s matrix is created (or updated) by setting the corresponding
        (subject_index, object_index) cell to True.

        Args:
            triple (tuple): A triple (subject, predicate, object) to be added.
        """
        s, p, o = triple
        s_index = self._get_node_index(s)
        o_index = self._get_node_index(o)

        # Create the predicate's matrix with the current preallocated capacity if it does not exist.
        if p not in self.pred_matrices:
            self.pred_matrices[p] = dok_matrix((self.matrix_capacity, self.matrix_capacity), dtype=bool)
        else:
            # Ensure the predicate matrix can accommodate the current capacity.
            if self.pred_matrices[p].shape[0] < self.matrix_capacity:
                self._expand_matrices_to(self.matrix_capacity)

        # Mark the presence of the triple in the predicate's matrix.
        self.pred_matrices[p][s_index, o_index] = True

    def remove(self, triple):
        """
        Remove a triple (subject, predicate, object) from the graph.

        If the triple exists in the predicate’s matrix, it is deleted.

        Args:
            triple (tuple): A triple (subject, predicate, object) to be removed.
        """
        s, p, o = triple
        if p not in self.pred_matrices:
            return  # The predicate is not present.
        try:
            s_index = self.node2index[s]
            o_index = self.node2index[o]
        except KeyError:
            # One or both nodes are not present in the graph.
            return

        mat = self.pred_matrices[p]
        # Delete the entry corresponding to the triple, if it exists.
        if (s_index, o_index) in mat:
            del mat[s_index, o_index]

    def __iter__(self):
        """
        Iterate over all triples in the graph.

        Iterates through each predicate’s matrix and yields triples for each nonzero entry.
        """
        for pred, mat in self.pred_matrices.items():
            for (i, j) in mat.keys():
                yield (self.index2node[i], pred, self.index2node[j])

    def __contains__(self, triple) -> bool:
        """
        Check whether a triple (subject, predicate, object) exists in the graph.

        Args:
            triple (tuple): A triple (subject, predicate, object) to be checked.

        Returns:
            bool: True if the triple exists, False otherwise.
        """
        s, p, o = triple
        if p not in self.pred_matrices or s not in self.node2index or o not in self.node2index:
            return False
        s_index = self.node2index[s]
        o_index = self.node2index[o]
        return (s_index, o_index) in self.pred_matrices[p]

    def union(self, other: "SparseMatrixGraph") -> "SparseMatrixGraph":
        """
        Compute the union of this graph with another SparseMatrixGraph.

        A new SparseMatrixGraph instance is created that contains all triples
        from both graphs. The process involves:

        1. Merging the node sets from both graphs and building a unified node mapping.
        2. Merging the sparse matrices for each predicate present in either graph using
           the unified mapping.

        Args:
            other (SparseMatrixGraph): Another graph to union with.

        Returns:
            SparseMatrixGraph: A new graph representing the union of both graphs.
        """
        # Merge the node sets from both graphs.
        nodes_self = set(self.node2index.keys())
        nodes_other = set(other.node2index.keys())
        all_nodes = nodes_self.union(nodes_other)

        # Initialize a new graph with the default preallocation capacity.
        union_graph = SparseMatrixGraph()
        for node in all_nodes:
            union_graph._get_node_index(node)

        # Determine the full set of predicates from both graphs.
        predicates = set(self.pred_matrices.keys()).union(other.pred_matrices.keys())
        for pred in predicates:
            # Create an empty matrix for the predicate using union_graph's capacity.
            union_mat = dok_matrix((union_graph.matrix_capacity, union_graph.matrix_capacity), dtype=bool)

            # Add triples from self.
            if pred in self.pred_matrices:
                for (i, j) in self.pred_matrices[pred].keys():
                    s = self.index2node[i]
                    o = self.index2node[j]
                    new_i = union_graph.node2index[s]
                    new_j = union_graph.node2index[o]
                    union_mat[new_i, new_j] = True

            # Add triples from other.
            if pred in other.pred_matrices:
                for (i, j) in other.pred_matrices[pred].keys():
                    s = other.index2node[i]
                    o = other.index2node[j]
                    new_i = union_graph.node2index[s]
                    new_j = union_graph.node2index[o]
                    union_mat[new_i, new_j] = True

            # Save the merged matrix for this predicate.
            union_graph.pred_matrices[pred] = union_mat

        return union_graph

    def __or__(self, other: "SparseMatrixGraph") -> "SparseMatrixGraph":
        """
        Overload the '|' operator to compute the union of two SparseMatrixGraph instances.
        """
        return self.union(other)


# Register the SparseMatrixGraph as an rdflib Graph plugin.
# This allows users to select the sparse matrix implementation when creating graphs:
#   g = Graph("sparse_matrix")
register('sparse_matrix', Store, __name__, 'SparseMatrixGraph')

# ======================================================================
# === Example Usage (for testing and demonstration purposes) =========
# ======================================================================
if __name__ == "__main__":
    # Create two sample graphs.
    g1 = Graph(store="sparse_matrix")
    g2 = Graph(store="sparse_matrix")

    # Define some example nodes (subjects/objects) and predicates.
    s1 = URIRef("http://example.org/subject1")
    s2 = URIRef("http://example.org/subject2")
    p_knows = URIRef("http://xmlns.com/foaf/0.1/knows")
    o1 = URIRef("http://example.org/object1")
    o2 = URIRef("http://example.org/object2")

    # Add triples to graph 1.
    g1.add((s1, p_knows, o1))
    g1.add((s1, p_knows, o2))

    # Add triples to graph 2.
    g2.add((s2, p_knows, o1))
    g2.add((s2, p_knows, o2))

    # Compute the union using the overloaded '|' operator.
    union_graph = g1 | g2

    # Print out all triples in the union graph.
    print("Triples in the union graph:")
    for triple in union_graph:
        print(triple)

    g1.parse("https://brickschema.org/schema/1.4/Brick.ttl", format="turtle")
    print("Triples in g1:")
    for triple in g1:
        print(triple)
