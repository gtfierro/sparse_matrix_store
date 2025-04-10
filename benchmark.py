import time
from rdflib import Graph, URIRef
from store import SparseMatrixGraph

def benchmark_graph_loading(graph_type, graph_name):
    """
    Benchmark the loading of the Brick ontology into a specified graph type.

    Args:
        graph_type (Graph): The graph class to instantiate.
        graph_name (str): A name for the graph type being benchmarked.
    """
    print(f"Benchmarking {graph_name}...")

    # Initialize the graph.
    g = graph_type()

    # Measure the time taken to parse the Brick ontology.
    start_time = time.time()
    g.parse("https://brickschema.org/schema/1.4/Brick.ttl", format="turtle")
    elapsed_time = time.time() - start_time

    print(f"Time taken to load Brick ontology into {graph_name}: {elapsed_time:.2f} seconds")
    print(f"Number of triples in {graph_name}: {len(g)}")

def main():
    # Benchmark the default RDFlib Graph.
    benchmark_graph_loading(Graph, "RDFlib Graph")

    # Benchmark the SparseMatrixGraph.
    benchmark_graph_loading(lambda: Graph(store="sparse_matrix"), "SparseMatrixGraph")

if __name__ == "__main__":
    main()
