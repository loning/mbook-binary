"""
Visualize D1-9 dependency graphs showing elimination of circular dependencies
"""

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import numpy as np


def create_old_dependency_graph():
    """Create original circular dependency graph"""
    G = nx.DiGraph()
    
    # 添加节点
    nodes = {
        "Axiom": (0, 3),
        "D1-1": (0, 2),
        "D1-5": (-1, 1),
        "T3-2": (1, 1),
        "T3-3": (0, 0)
    }
    
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)
    
    # 添加边（包含循环依赖）
    edges = [
        ("Axiom", "D1-1"),
        ("D1-1", "D1-5"),
        ("D1-1", "T3-2"),
        ("D1-5", "T3-2", "red"),  # 循环依赖
        ("T3-2", "D1-5", "red"),  # 循环依赖
        ("T3-2", "T3-3"),
        ("D1-5", "T3-3")
    ]
    
    for edge in edges:
        if len(edge) == 3:
            G.add_edge(edge[0], edge[1], color=edge[2])
        else:
            G.add_edge(edge[0], edge[1], color="black")
    
    return G


def create_new_dependency_graph():
    """Create new acyclic dependency graph"""
    G = nx.DiGraph()
    
    # 添加节点
    nodes = {
        "Axiom": (0, 4),
        "SelfRef": (-1, 3),
        "InfoDist": (-2, 2),
        "PatternRec": (1, 3),
        "D1-9-Measure": (-2, 1),
        "D1-9-Observer": (2, 1),
        "Interaction": (0, 0),
        "D1-5": (-1, -1),
        "T3-2": (1, -1)
    }
    
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)
    
    # 添加边（无循环）
    edges = [
        ("Axiom", "SelfRef"),
        ("SelfRef", "InfoDist"),
        ("SelfRef", "PatternRec"),
        ("InfoDist", "D1-9-Measure"),
        ("PatternRec", "D1-9-Observer"),
        ("D1-9-Measure", "Interaction"),
        ("D1-9-Observer", "Interaction"),
        ("Interaction", "D1-5"),
        ("Interaction", "T3-2")
    ]
    
    for edge in edges:
        G.add_edge(edge[0], edge[1], color="green")
    
    return G


def plot_comparison():
    """Plot comparison graph"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：原有循环依赖
    G_old = create_old_dependency_graph()
    pos_old = nx.get_node_attributes(G_old, 'pos')
    
    ax1.set_title("Original Definition Structure (With Circular Dependencies)", fontsize=14, fontweight='bold')
    
    # 绘制节点
    nx.draw_networkx_nodes(G_old, pos_old, ax=ax1, 
                          node_color='lightcoral', 
                          node_size=1500, alpha=0.9)
    
    # 绘制边
    edge_colors_old = [G_old[u][v]['color'] for u, v in G_old.edges()]
    nx.draw_networkx_edges(G_old, pos_old, ax=ax1,
                          edge_color=edge_colors_old,
                          width=2, alpha=0.7,
                          connectionstyle="arc3,rad=0.1",
                          arrowsize=20, arrowstyle='->')
    
    # 绘制标签
    nx.draw_networkx_labels(G_old, pos_old, ax=ax1, font_size=10)
    
    # Add circular dependency annotation
    ax1.text(0, 0.5, "Circular Dependency!", fontsize=12, color='red',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 右图：新的无循环结构
    G_new = create_new_dependency_graph()
    pos_new = nx.get_node_attributes(G_new, 'pos')
    
    ax2.set_title("D1-9 New Definition Structure (DAG - No Cycles)", fontsize=14, fontweight='bold')
    
    # 节点颜色
    node_colors = []
    for node in G_new.nodes():
        if node == "Axiom":
            node_colors.append('gold')
        elif "D1-9" in node:
            node_colors.append('lightgreen')
        elif node == "Interaction":
            node_colors.append('lightblue')
        else:
            node_colors.append('lightgray')
    
    nx.draw_networkx_nodes(G_new, pos_new, ax=ax2,
                          node_color=node_colors,
                          node_size=1500, alpha=0.9)
    
    # 绘制边
    nx.draw_networkx_edges(G_new, pos_new, ax=ax2,
                          edge_color='green',
                          width=2, alpha=0.7,
                          connectionstyle="arc3,rad=0.05",
                          arrowsize=20, arrowstyle='->')
    
    # Draw labels
    labels_new = {
        "Axiom": "Unique Axiom",
        "SelfRef": "Self-Reference",
        "InfoDist": "Info Distinction",
        "PatternRec": "Pattern Recogn.",
        "D1-9-Measure": "Measurement Def",
        "D1-9-Observer": "Observer Def",
        "Interaction": "Interaction",
        "D1-5": "D1-5(Compat)",
        "T3-2": "T3-2(Compat)"
    }
    nx.draw_networkx_labels(G_new, pos_new, labels_new, ax=ax2, font_size=9)
    
    # Add DAG annotation
    ax2.text(0, -2, "✓ DAG Structure\nNo Circular Dependencies", fontsize=12, color='green',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 设置轴属性
    for ax in [ax1, ax2]:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2.5, 4.5)
        ax.axis('off')
    
    plt.suptitle("D1-9: Measurement-Observer Separation Definition - Circular Dependency Elimination", 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('D1_9_dependency_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Dependency comparison graph saved as D1_9_dependency_comparison.png")


def plot_derivation_chain():
    """Plot derivation chain"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create derivation chain graph
    G = nx.DiGraph()
    
    # Define hierarchical structure
    levels = {
        0: ["Unique Axiom:\nSelf-referential\ncomplete system\nmust increase entropy"],
        1: ["Self-reference\nRequirement", "Entropy Increase\nRequirement"],
        2: ["Information\nDistinction\nCapability", "Pattern Recognition\nNeed", "System\nEvolution"],
        3: ["State\nTransformation", "Encoding\nMechanism", "Subsystem\nStructure"],
        4: ["Measurement\nProcess\n(D1-9.1)", "Observer\nSystem\n(D1-9.2)"],
        5: ["Interaction\n(D1-9.3)"],
        6: ["Quantum\nMeasurement\n(T3-2)", "Observer\nEmergence\n(D1-5)"]
    }
    
    # 添加节点并设置位置
    pos = {}
    y_spacing = 1.5
    
    for level, nodes in levels.items():
        x_spacing = 6.0 / (len(nodes) + 1)
        for i, node in enumerate(nodes):
            x = -3 + (i + 1) * x_spacing
            y = 6 - level * y_spacing
            G.add_node(node)
            pos[node] = (x, y)
    
    # Add edges
    edges = [
        ("Unique Axiom:\nSelf-referential\ncomplete system\nmust increase entropy", "Self-reference\nRequirement"),
        ("Unique Axiom:\nSelf-referential\ncomplete system\nmust increase entropy", "Entropy Increase\nRequirement"),
        ("Self-reference\nRequirement", "Information\nDistinction\nCapability"),
        ("Self-reference\nRequirement", "Pattern Recognition\nNeed"),
        ("Entropy Increase\nRequirement", "System\nEvolution"),
        ("Information\nDistinction\nCapability", "State\nTransformation"),
        ("Information\nDistinction\nCapability", "Encoding\nMechanism"),
        ("Pattern Recognition\nNeed", "Subsystem\nStructure"),
        ("State\nTransformation", "Measurement\nProcess\n(D1-9.1)"),
        ("Encoding\nMechanism", "Measurement\nProcess\n(D1-9.1)"),
        ("Subsystem\nStructure", "Observer\nSystem\n(D1-9.2)"),
        ("Encoding\nMechanism", "Observer\nSystem\n(D1-9.2)"),
        ("Measurement\nProcess\n(D1-9.1)", "Interaction\n(D1-9.3)"),
        ("Observer\nSystem\n(D1-9.2)", "Interaction\n(D1-9.3)"),
        ("Interaction\n(D1-9.3)", "Quantum\nMeasurement\n(T3-2)"),
        ("Interaction\n(D1-9.3)", "Observer\nEmergence\n(D1-5)")
    ]
    
    G.add_edges_from(edges)
    
    # Node colors
    node_colors = []
    for node in G.nodes():
        if "Unique Axiom" in node:
            node_colors.append('#FFD700')  # Gold
        elif "D1-9" in node:
            node_colors.append('#90EE90')  # Light green
        elif "Interaction" in node:
            node_colors.append('#87CEEB')  # Sky blue
        elif any(x in node for x in ["T3-2", "D1-5"]):
            node_colors.append('#FFB6C1')  # Light pink
        else:
            node_colors.append('#E6E6FA')  # Lavender
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=2500, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          width=2, alpha=0.6, arrows=True,
                          arrowsize=20, arrowstyle='->', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    
    # Add title and description
    ax.set_title("D1-9 Complete Derivation Chain: From Unique Axiom to Measurement-Observer Separation", 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add explanatory text
    ax.text(-3.5, -3, 
           "Key Properties:\n"
           "• Independent measurement and observer definitions\n"
           "• No circular dependencies (DAG structure)\n"
           "• Linear derivation from unique axiom\n"
           "• Maintains functional completeness",
           fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 7)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('D1_9_derivation_chain.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Derivation chain graph saved as D1_9_derivation_chain.png")


if __name__ == "__main__":
    print("Generating D1-9 dependency relationship visualization...")
    
    # Generate comparison graph
    plot_comparison()
    
    # Generate derivation chain graph
    plot_derivation_chain()
    
    print("\nVisualization complete!")
    print("- D1_9_dependency_comparison.png: Circular dependency elimination comparison")
    print("- D1_9_derivation_chain.png: Complete derivation chain display")