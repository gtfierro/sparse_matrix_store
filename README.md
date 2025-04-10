# SparseMatrixStore

SparseMatrixStore is an RDF store implementation using a sparse matrix backend. It leverages the `rdflib` library to provide efficient storage and querying of RDF triples using sparse matrices. This approach is particularly useful for handling large datasets with many nodes but relatively few connections.

## Features

- **Sparse Matrix Backend**: Utilizes `scipy.sparse` for efficient storage of RDF triples.
- **Preallocated Capacity**: Allows for preallocation of matrix capacity to optimize performance.
- **Integration with RDFlib**: Seamlessly integrates with RDFlib's `Graph` class for RDF data manipulation.

## How It Works

### Core Components

- **Node Indexing**: Nodes (subjects and objects) are mapped to unique integer indices for efficient matrix operations.
- **Predicate Matrices**: Each predicate is associated with a sparse matrix (`dok_matrix`) that records the presence of triples.
- **Dynamic Expansion**: The store dynamically expands its capacity when the number of nodes exceeds the current allocation.

### Operations

- **Add Triples**: Adds RDF triples to the store by updating the corresponding predicate matrix.
- **Remove Triples**: Removes RDF triples by deleting entries from the predicate matrix.
- **Query Triples**: Supports pattern-based querying to retrieve matching triples.

## Usage

### Example

```python
from rdflib import URIRef
from rdflib.graph import Graph
from store import SparseMatrixStore

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
```

## Benchmarking

The `benchmark.py` script provides a way to benchmark the performance of the SparseMatrixStore against the default RDFlib Graph. It measures the time taken to load the Brick ontology and execute SPARQL queries.

### Running Benchmarks

To run the benchmarks, execute the following command:

```bash
python benchmark.py
```

This will output the time taken to load the ontology and execute queries for both the default RDFlib Graph and the SparseMatrixStore.

## License

This project is licensed under the MIT License.
