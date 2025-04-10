import time
from rdflib import Graph, URIRef

# from store import SparseMatrixG
import store


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

    print(
        f"[{graph_name}] Time taken to load Brick ontology into {graph_name}: {elapsed_time:.2f} seconds"
    )
    print(f"[{graph_name}] Number of triples in {graph_name}: {len(g)}")
    return g


def benchmark_graph_query(graph, query, graph_name):
    """
    Benchmark the execution of a SPARQL query on a graph.

    Args:
        graph (Graph): The graph to query.
        query (str): The SPARQL query string.
    """
    start_time = time.time()
    results = graph.query(query)
    elapsed_time = time.time() - start_time

    print(f"[{graph_name}] taken to execute query: {elapsed_time:.2f} seconds")
    print(f"[{graph_name}] Number of results: {len(results)}")
    return results


def main():
    # Benchmark the default RDFlib Graph.
    g1 = benchmark_graph_loading(Graph, "RDFlib Graph")

    # Benchmark the SparseMatrixGraph.
    g2 = benchmark_graph_loading(
        lambda: Graph(store="sparse_matrix"), "SparseMatrixGraph"
    )

    # Example SPARQL query to benchmark.
    query = """
    SELECT ?path WHERE {
            {
            ?shape sh:property ?prop .
            ?prop sh:path ?path .
            }
            UNION
            {
            ?path a owl:ObjectProperty .
            }
             FILTER (!isBlank(?path))
    }"""
    # Benchmark the query on the RDFlib Graph.
    r1 = benchmark_graph_query(g1, query, "RDFlib Graph")
    print(f"# results RDFlib Graph: {len(r1)}")
    # Benchmark the query on the SparseMatrixGraph.
    r2 = benchmark_graph_query(g2, query, "SparseMatrixGraph")
    print(f"# results SparseMatrixGraph: {len(r2)}")


if __name__ == "__main__":
    main()
