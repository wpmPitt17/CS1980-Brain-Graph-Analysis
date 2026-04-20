# connectome_graph.py
# Author: Rhonda Ojongmboh
#
# Generates a connectome network graph from Ranger importance scores
# using Cytoscape via py4cytoscape. ROIs are treated as Craddock parcel
# indices and exported as numeric labels.
#
# Requires: Cytoscape desktop running locally, py4cytoscape, pandas
#
# Usage:
#   python connectome_graph.py -i ranger_out.importance -o connectome_graph.pdf
#   python connectome_graph.py -i ranger_out.importance -o connectome_graph.pdf --top 50
#   python connectome_graph.py -i ranger_out.importance -o connectome_graph.pdf --top 75 --condition ASD

import argparse
import os
import pandas as pd
import py4cytoscape as p4c

# ------------------------------------------------------------
def get_label(roi):
    """Node label for Craddock parcels."""
    return f"ROI{roi}"

# ------------------------------------------------------------
# Load importance file
# ------------------------------------------------------------
def load_importance(filepath, top_n):
    """
    Loads ranger_out.importance file.
    Format per line: ROIx_ROIy: score
    Returns DataFrame of top_n edges sorted by importance descending.
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, val = line.split(': ')
            score = float(val)
            a, b = key.split('_')
            roi_a = int(a.replace('ROI', ''))
            roi_b = int(b.replace('ROI', ''))
            records.append({'source': roi_a, 'target': roi_b, 'importance': score})

    df = pd.DataFrame(records).sort_values('importance', ascending=False).head(top_n)
    return df.reset_index(drop=True)

# ------------------------------------------------------------
# Build node and edge tables for Cytoscape
# ------------------------------------------------------------
def build_tables(edges_df):
    """
    Builds node and edge DataFrames for py4cytoscape.
    Nodes: id, label, node_color, degree, total_importance
    Edges: source, target, importance, width (scaled 1-10)
    """
    node_ids = pd.unique(edges_df[['source', 'target']].values.ravel())

    degree = {}
    total_imp = {}
    for _, row in edges_df.iterrows():
        for n in [row['source'], row['target']]:
            degree[n]    = degree.get(n, 0) + 1
            total_imp[n] = total_imp.get(n, 0.0) + row['importance']

    nodes = []
    for roi in node_ids:
        nodes.append({
            'id':               str(roi),
            'label':            get_label(roi),
            'node_color':       '#4A90D9',
            'degree':           degree.get(roi, 0),
            'total_importance': round(total_imp.get(roi, 0.0), 4),
        })
    nodes_df = pd.DataFrame(nodes)

    min_imp   = edges_df['importance'].min()
    max_imp   = edges_df['importance'].max()
    imp_range = max_imp - min_imp if max_imp != min_imp else 1.0

    edges_out = edges_df.copy()
    edges_out['source'] = edges_out['source'].astype(str)
    edges_out['target'] = edges_out['target'].astype(str)
    edges_out['width']  = 1 + 9 * (edges_out['importance'] - min_imp) / imp_range

    return nodes_df, edges_out

# ------------------------------------------------------------
# Apply visual style in Cytoscape
# ------------------------------------------------------------
def apply_style(style_name):
    existing = p4c.get_visual_style_names()
    if style_name in existing:
        p4c.delete_visual_style(style_name)

    p4c.create_visual_style(style_name)

    # Node label
    p4c.set_node_label_mapping('label', style_name=style_name)

    # Node color: passthrough from node_color column
    p4c.set_node_color_mapping(
        table_column='node_color',
        mapping_type='passthrough',
        style_name=style_name
    )

    # Node size by degree: range 30-90px
    p4c.set_node_size_mapping(
        table_column='degree',
        table_column_values=[1, 5, 10],
        sizes=[30, 55, 90],
        mapping_type='continuous',
        style_name=style_name
    )

    # Node border
    p4c.set_node_border_width_default(2, style_name=style_name)
    p4c.set_node_border_color_default('#444444', style_name=style_name)

    # Node label font
    p4c.set_node_font_size_default(9, style_name=style_name)

    # Edge width: passthrough from width column
    p4c.set_edge_line_width_mapping(
        table_column='width',
        mapping_type='passthrough',
        style_name=style_name
    )

    # Edge appearance
    p4c.set_edge_color_default('#555555', style_name=style_name)
    p4c.set_edge_opacity_default(160, style_name=style_name)

    # White background
    p4c.set_background_color_default('#FFFFFF', style_name=style_name)

    p4c.set_visual_style(style_name)
    print(f"  Visual style '{style_name}' applied.")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Connectome network graph from Ranger importance scores via Cytoscape.'
    )
    parser.add_argument('-i', '--input',     required=True,
                        help='Path to ranger_out.importance file')
    parser.add_argument('-o', '--output',    required=True,
                        help='Output PDF path (e.g. connectome_graph.pdf)')
    parser.add_argument('--top',             type=int, default=30,
                        help='Number of top edges to include (default: 30)')
    parser.add_argument('--condition',       type=str, default='Condition',
                        help='Disease/condition label for graph title (default: Condition)')
    parser.add_argument('--layout',          type=str, default='force-directed',
                        help='Cytoscape layout name (default: force-directed)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    out_path = os.path.abspath(args.output)
    if not out_path.endswith('.pdf'):
        out_path += '.pdf'

    print(f"Loading importance scores from: {args.input}")
    edges_df = load_importance(args.input, args.top)
    print(f"  Loaded top {len(edges_df)} edges.")
    print(f"  Score range: {edges_df['importance'].min():.4f} - {edges_df['importance'].max():.4f}")

    nodes_df, edges_df = build_tables(edges_df)
    print(f"  {len(nodes_df)} unique ROIs in graph.")

    print("Connecting to Cytoscape (make sure Cytoscape is running)...")
    p4c.cytoscape_ping()
    p4c.cytoscape_version_info()

    network_title = f"{args.condition} Connectome - Top {args.top} Edges"
    print(f"Creating network: '{network_title}'")
    network_suid = p4c.create_network_from_data_frames(
        nodes=nodes_df,
        edges=edges_df,
        source_id_list='source',
        target_id_list='target',
        node_id_list='id',
        title=network_title,
    )
    print(f"  Network SUID: {network_suid}")

    print(f"Applying layout: {args.layout}")
    p4c.layout_network(args.layout)

    style_name = f"{args.condition}_connectome_style"
    print("Applying visual style...")
    apply_style(style_name)

    print(f"Exporting PDF to: {out_path}")
    p4c.export_image(out_path, type='PDF', network=network_suid)
    print(f"Done. Graph saved to: {out_path}")

    # Summary of top hub regions
    top_hubs = nodes_df.sort_values('total_importance', ascending=False).head(5)
    print("\nTop 5 hub ROIs by aggregated importance:")
    for _, row in top_hubs.iterrows():
        roi = int(row['id'])
        print(f"  ROI{roi:>3} | degree={row['degree']} "
              f"| score={row['total_importance']:.4f}")


if __name__ == '__main__':
    main()
