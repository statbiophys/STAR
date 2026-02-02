import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Levenshtein import distance
import matplotlib.patches as mpatches
"""
        Class to visualize clusters of sequences using a graph representation.
        It uses Levenshtein distance to determine edges between nodes (sequences).
        The nodes can be colored and sized based on features from the DataFrame.
        Example usage:
        from Graph_cluster import ClusterGraph
        graph = ClusterGraph(
        df=df_sub, your dataframe subset for a single cluster
        feature="junction_aa", feature to be used as node label
        color_feature="exp_mouse", feature to be used for node color
        size_feature="Sharing_size" , feature to be used for node size
    )

        graph.plot_cluster(
            df_sub, # your dataframe subset for a single cluster
            distance_threshold=1, # max Levenshtein distance for creating an edge
            title = "Cluster {}".format(cl), # title of the plot
            
        )
"""

class ClusterGraph:
    def __init__(
        self,
        df: pd.DataFrame,
        feature: str = "junction_aa",
        color_feature: str = None,
        size_feature: str = None,
        palette: str = "hls",
        color_map: dict = None,
    ):
        """
        Initializes the ClusterGraph object.

        Parameters:
        df (pd.DataFrame): DataFrame containing clustered sequences.
        feature (str): Column containing sequence data (default: "junction_aa").
        color_feature (str): Column used to color nodes. If None, nodes will be gray.
        size_feature (str): Column used to scale node sizes. If None, default size is used.
        palette (str): Seaborn palette to use if color_map is not provided.
        color_map (dict): Optional custom color map.
        """
        self.df = df
        self.feature = feature
        self.color_feature = color_feature
        self.size_feature = size_feature
        self.palette = palette

        # Auto-generate color map if not provided
        if color_feature is not None and color_map is None:
            unique_vals = df[color_feature].dropna().unique()
            colors = sns.color_palette(palette, len(unique_vals))
            self.color_map = dict(zip(unique_vals, colors))
        else:
            self.color_map = color_map

    def _lev_distance(self, s1, s2):
        """Compute the Levenshtein distance between two sequences."""
        return distance(s1, s2)
    
    def plot_cluster(
        self,
        df_sub: pd.DataFrame,
        show_legend: bool = True,
        distance_threshold: int = 1,
        title: str = None,
        min_size: int = 1  # <-- NEW: minimum connected component size to display
    ):
        """
        Plots a cluster from a given DataFrame with edges for Levenshtein distance <= threshold.
        Disconnected components are laid out in a grid. Filters out components smaller than min_size.
        """
        import math

        df_sub = df_sub.reset_index(drop=True)
        G = nx.Graph()
        G.add_nodes_from(df_sub.index)

        # Build edges based on Levenshtein distance
        for i in range(len(df_sub)):
            for j in range(i + 1, len(df_sub)):
                if self._lev_distance(df_sub[self.feature][i], df_sub[self.feature][j]) <= distance_threshold:
                    G.add_edge(i, j)

        # Remove small connected components
        if min_size > 1:
            small_components = [comp for comp in nx.connected_components(G) if len(comp) < min_size]
            for comp in small_components:
                G.remove_nodes_from(comp)

        # Set node sizes
        if self.size_feature is not None:
            node_sizes = df_sub.loc[list(G.nodes)][self.size_feature].fillna(0).apply(lambda x: 2 * x + 30).tolist()
        else:
            node_sizes = [50] * len(G.nodes())

        # Set node colors
        if self.color_feature is not None and self.color_map is not None:
            node_colors = df_sub.loc[list(G.nodes)][self.color_feature].map(self.color_map).tolist()
        else:
            node_colors = ["gray"] * len(G.nodes())

        # Layout each connected component separately in a grid
        components = list(nx.connected_components(G))
        cols = int(math.ceil(math.sqrt(len(components))))
        pos = {}
        spacing = 10

        for idx, comp in enumerate(components):
            subG = G.subgraph(comp)
            sub_pos = nx.spring_layout(subG, seed=42, k=0.8, scale=4)
            row, col = divmod(idx, cols)
            dx, dy = col * spacing, -row * spacing
            for node, (x, y) in sub_pos.items():
                pos[node] = (x + dx, y + dy)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))  # larger figure
        plt.rcParams.update({
            'font.size': 16,        # general font size
            'axes.titlesize': 24,   # title font size
            'axes.labelsize': 18,   # label font size (if used)
            'legend.fontsize': 14,  # legend font size
        })

        # Draw graph on the axes
        nx.draw(
            G,
            pos,
            node_size=node_sizes,
            width=1.2,
            edge_color="black",
            node_color=node_colors,
            with_labels=True,
            ax=ax
        )

        # Add legend if enabled
        if show_legend and self.color_feature and self.color_map:
            legend_handles = [
                mpatches.Patch(color=color, label=str(label))
                for label, color in self.color_map.items()
            ]
            ax.legend(
                handles=legend_handles,
                frameon=False,
                title=self.color_feature,
                title_fontsize=18,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
            )

        # Larger, bold title
        ax.set_title(
            title if title is not None else f"Cluster (LD ≤ {distance_threshold})",
            fontsize=26,
            pad=20
        )

        ax.set_axis_off()
        plt.tight_layout()
        return fig


