import networkx as nx
import plotly.graph_objects as go
from collections import Counter
from urllib.parse import urlparse
from data_collector import OpenAlexAPI
from matplotlib import pyplot as plt
import numpy as np

def extract_id(openalex_url):
    """Extract the ID from an OpenAlex URL (e.g., W3159481202 from https://openalex.org/W3159481202)."""
    return urlparse(openalex_url).path.split('/')[-1]

def build_citation_graph_v1(root_id, data, root_title=None):
    """Build a directed citation graph with edges from cited papers to the root paper."""
    G = nx.DiGraph()
    root_id = extract_id(root_id)

    # Step 1: Add root paper node
    root_label = root_title if root_title else f"Paper {root_id}"
    G.add_node(root_id, label=root_label, type='root')

    # Step 2: Add cited papers as nodes
    cited_papers = set(data.keys())
    for pid in cited_papers:
        pid_extracted = extract_id(pid)
        title = data[pid].get('title', f"Paper {pid_extracted}")
        G.add_node(pid_extracted, label=title, type='cited')

    # Step 3: Add edges (from cited papers to root paper)
    edges = []
    for pid in cited_papers:
        pid_extracted = extract_id(pid)
        if pid_extracted != root_id:  # Avoid self-loop if root is in cited papers
            G.add_edge(pid_extracted, root_id)
            edges.append((pid_extracted, root_id))

    print(f"DEBUG: Generated {len(edges)} edges")  # Debugging output
    return G, edges


def build_citation_graph(root_id, data, root_title=None, embedding_model=None):
    """
    Build a directed citation graph with edges from cited papers to root and related works to cited papers.
    Optionally pre-compute node embeddings.
    
    Args:
        root_id: OpenAlex ID of the root paper.
        data: Citation data from OpenAlex (from citations.json).
        root_title: Title of the root paper (optional).
        embedding_model: SentenceTransformer model for embedding node labels (optional).
    
    Returns:
        G: NetworkX DiGraph.
        edges: List of edge tuples.
    """
    G = nx.DiGraph()
    root_id = extract_id(root_id)

    # Step 1: Add root paper node
    root_label = root_title if root_title else f"Paper {root_id}"
    root_attrs = {'label': root_label, 'type': 'root', 'cited_by_count': data.get(root_id, {}).get('cited_by_count', 0)}
    if embedding_model:
        root_attrs['embedding'] = embedding_model.embed_query(root_label).tolist()
    G.add_node(root_id, **root_attrs)

    # Step 2: Add cited papers as nodes
    cited_papers = set(data.keys())
    for pid in cited_papers:
        pid_extracted = extract_id(pid)
        title = data[pid].get('title', f"Paper {pid_extracted}")
        attrs = {
            'label': title,
            'type': 'cited',
            'cited_by_count': data[pid].get('cited_by_count', 0),
            'publication_year': data[pid].get('publication_year', None)
        }
        if embedding_model:
            attrs['embedding'] = embedding_model.embed_query(title).tolist()
        G.add_node(pid_extracted, **attrs)

    # Step 3: Add related works as nodes
    related_works = set()
    for pid in cited_papers:
        for rw in data[pid].get('related_works', []):
            rw_id = extract_id(rw)
            if rw_id not in cited_papers and rw_id != root_id:
                related_works.add(rw_id)
                attrs = {'label': f"Ref {rw_id}", 'type': 'related', 'cited_by_count': 0}
                if embedding_model:
                    attrs['embedding'] = embedding_model.embed_query(attrs['label']).tolist()
                G.add_node(rw_id, **attrs)

    # Step 4: Add edges with weights
    edges = []
    for pid in cited_papers:
        pid_extracted = extract_id(pid)
        if pid_extracted != root_id:
            weight = data[pid].get('cited_by_count', 1) / 100.0  # Normalize weight
            G.add_edge(pid_extracted, root_id, weight=weight)
            edges.append((pid_extracted, root_id))
    for pid in cited_papers:
        pid_extracted = extract_id(pid)
        for rw in data[pid].get('related_works', []):
            rw_id = extract_id(rw)
            if rw_id in related_works:
                G.add_edge(rw_id, pid_extracted, weight=1.0)  # Default weight for related works
                edges.append((rw_id, pid_extracted))

    print(f"DEBUG: Generated {len(edges)} edges (cited→root: {len(cited_papers) - (1 if root_id in [extract_id(p) for p in cited_papers] else 0)}, related→cited: {sum(len(data[pid].get('related_works', [])) for pid in cited_papers)})")
    return G, edges

def graph_retrieval(graph, query_text, top_k=3, max_hops=2, citation_data=None, embedding_model=None):
    """
    Retrieve top-K relevant nodes from the citation graph using semantic similarity and graph metrics.
    
    Args:
        graph: NetworkX DiGraph containing papers and citations.
        query_text: User query string.
        top_k: Number of nodes to return.
        max_hops: Maximum hops for neighbor expansion.
        citation_data: Citation data from OpenAlex (from citations.json) for metadata.
        embedding_model: SentenceTransformer model for embedding query and node labels.
    
    Returns:
        List of dictionaries containing node IDs, attributes, and relevance scores.
    """
    # Embed query
    query_embedding = embedding_model.embed_query(query_text) if embedding_model else None
    node_scores = []
    
    # Score nodes based on semantic similarity and citation count
    for node in graph.nodes(data=True):
        node_id, node_data = node
        label = node_data.get('label', '')
        # Semantic similarity
        if embedding_model and 'embedding' in node_data:
            node_embedding = np.array(node_data['embedding'])
            similarity = np.dot(query_embedding, node_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
        else:
            similarity = 1.0 if query_text.lower() in label.lower() else 0.0
        # Citation count (normalized)
        cited_by_count = node_data.get('cited_by_count', 0)
        citation_score = min(cited_by_count / 100, 1.0)  # Cap at 100 citations
        # Combined score
        score = 0.6 * similarity + 0.3 * citation_score
        node_scores.append((node_id, score))
    
    # Select top-K initial nodes
    node_scores = sorted(node_scores, key=lambda x: x[1], reverse=True)
    relevant_nodes = [node_id for node_id, _ in node_scores[:top_k]]
    
    # Expand to neighbors (directed traversal)
    expanded_nodes = set(relevant_nodes)
    hop_distances = {node: 0 for node in relevant_nodes}
    for hop in range(max_hops):
        current_nodes = list(expanded_nodes)
        for node in current_nodes:
            # Follow successors (cited papers) and predecessors (citing papers)
            for neighbor in list(graph.successors(node)) + list(graph.predecessors(node)):
                if neighbor not in expanded_nodes:
                    expanded_nodes.add(neighbor)
                    hop_distances[neighbor] = hop + 1
    
    # Rank expanded nodes
    expanded_papers = []
    for node in expanded_nodes:
        node_data = graph.nodes[node]
        label = node_data.get('label', '')
        # Semantic similarity
        if embedding_model and 'embedding' in node_data:
            node_embedding = np.array(node_data['embedding'])
            similarity = np.dot(query_embedding, node_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
        else:
            similarity = 1.0 if query_text.lower() in label.lower() else 0.0
        # Citation count
        cited_by_count = node_data.get('cited_by_count', 0)
        citation_score = min(cited_by_count / 100, 1.0)
        # Hop distance penalty
        hop_distance = hop_distances.get(node, max_hops)
        hop_score = 1.0 - (hop_distance / (max_hops + 1))  # Normalize to 0-1
        # Combined score
        relevance_score = 0.6 * similarity + 0.3 * citation_score + 0.1 * hop_score
        # Enrich with metadata from citation_data
        metadata = {
            'id': node,
            'label': label,
            'type': node_data.get('type'),
            'cited_by_count': cited_by_count,
            'publication_year': node_data.get('publication_year'),
            'relevance_score': relevance_score
        }
        if citation_data and node in citation_data:
            metadata.update({
                'title': citation_data[node].get('title'),
                'publication_year': citation_data[node].get('publication_year')
            })
        expanded_papers.append(metadata)
    
    # Sort by relevance and return top-K
    expanded_papers = sorted(expanded_papers, key=lambda x: x['relevance_score'], reverse=True)
    return expanded_papers[:top_k]

def fuse_and_rank_results(vector_results, graph_papers, top_n=5):
    fused_results = []

    for i in range(len(vector_results['documents'][0])):
        fused_results.append({
            "source": "vector_db",
            "content": vector_results['documents'][0][i],
            "metadata": vector_results['metadatas'][0][i],
            "relevance_score": vector_results['distances'][0][i]
        })

    for paper in graph_papers:
        fused_results.append({
            "source": "citation_graph",
            "content": f"Title: {paper.get('label', 'N/A')}",
            "metadata": {"openalex_id": paper.get('openalex_id', None)},
            "relevance_score": 0.7  # Slightly lower score for graph results in basic fusion
        })

    ranked_results = sorted(fused_results, key=lambda x: x['relevance_score']) # Lower score is better for distance
    ranked_results_without_score = [{k: v for k, v in res.items() if k != 'relevance_score'} for res in ranked_results]

    return ranked_results_without_score[:top_n]

def visualize_static(G, edges):
    """Visualize the graph using Matplotlib with visible edges."""
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    node_types = nx.get_node_attributes(G, 'type')

    plt.figure(figsize=(12, 8))
    node_colors = ['red' if node_types[n] == 'root' else 'lightblue' for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color=node_colors,
            font_size=8, font_weight='bold', arrows=True, arrowstyle='->', arrowsize=20)
    plt.title("Static Citation Graph")
    plt.show()

def visualize_interactive(G, edges, root_id):
    """Visualize the graph interactively using Plotly with visible edges."""
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    node_types = nx.get_node_attributes(G, 'type')

    # Edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{edge[0]} → {edge[1]}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#555'),
        hoverinfo='text',
        text=edge_text[::3],
        mode='lines'
    )

    # Node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(labels[node])
        node_colors.append('red' if node == root_id else 'lightblue')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(size=20, color=node_colors, line=dict(width=2))
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Citation Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    # fig.show()
    return fig

def main(root_id, data, root_title=None):
    """Main function to build and visualize the citation graph."""
    G, edges = build_citation_graph(root_id, data, root_title)
    
    # Print nodes and edges
    print("Nodes:")
    for node, attr in G.nodes(data=True):
        print(f"- {node}: {attr['label']} ({attr['type']})")
    print("\nEdges:")
    if not edges:
        print("No edges found in the graph.")
    for citing, cited in edges:
        print(f"- {citing} → {cited}")
    
    # Visualize the graph
    # visualize_static(G, edges)
    visualize_interactive(G, edges, extract_id(root_id))


if __name__ == "__main__":

    # paper = "Attention is all you need"
    # openalex_api = OpenAlexAPI(paper)
    # data = openalex_api.get_citations()
    import simplejson as json
    with open("citations.json", "r") as _file:
        data = json.load(_file)
    # main(
    #     root_id=openalex_api.query,
    #     data=data[openalex_api.query_alex_repsone.get('id', "root")],
    #     root_title=paper
    # )

    main(
        root_id=list(data.keys())[0],
        data=data[list(data.keys())[0]],
        root_title="Attention is all you need"
    )