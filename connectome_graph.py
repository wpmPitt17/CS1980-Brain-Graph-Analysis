# connectome_graph.py
# Author: Rhonda Ojongmboh
#
# Generates an annotated connectome network graph from Ranger importance scores
# using Cytoscape via py4cytoscape. Maps ROI indices to Harvard-Oxford atlas
# region names and exports the graph as a PDF.
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

# C++ engine -- built from connectome_engine.cpp via CMakeLists.txt
# Provides: run(), pearson_r(), parse_1D()
try:
    import connectome_engine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("Warning: connectome_engine not found. Build it first with CMakeLists.txt.")

# C++ engine -- built from connectome_engine.cpp via CMakeLists.txt
# Provides: run(), pearson_r(), parse_1D()
try:
    import connectome_engine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("Warning: connectome_engine not found. Build it first with CMakeLists.txt.")
import os
import pandas as pd
import py4cytoscape as p4c

# ------------------------------------------------------------
# Harvard-Oxford atlas (rois_ho, 112 regions)
# ROI index -> anatomical region name
# ------------------------------------------------------------
HO_LABELS = {
    0:  'Frontal Pole',                         1:  'Insular Cortex',
    2:  'Superior Frontal Gyrus',               3:  'Middle Frontal Gyrus',
    4:  'Inf. Frontal Gyrus (tri)',              5:  'Inf. Frontal Gyrus (oper)',
    6:  'Precentral Gyrus (L)',                  7:  'Precentral Gyrus',
    8:  'Temporal Pole',                         9:  'Sup. Temporal Gyrus (ant)',
    10: 'Sup. Temporal Gyrus (post)',            11: 'Mid. Temporal Gyrus (ant)',
    12: 'Mid. Temporal Gyrus (post)',            13: 'Mid. Temporal Gyrus (tO)',
    14: 'Inf. Temporal Gyrus (ant)',             15: 'Inf. Temporal Gyrus (post)',
    16: 'Inf. Temporal Gyrus (tO)',              17: 'Postcentral Gyrus',
    18: 'Sup. Parietal Lobule',                  19: 'Supramarginal Gyrus (ant)',
    20: 'Supramarginal Gyrus (post)',            21: 'Angular Gyrus',
    22: 'Lateral Occipital (sup)',               23: 'Lateral Occipital (inf)',
    24: 'Intracalcarine Cortex',                 25: 'Frontal Medial Cortex',
    26: 'Juxtapositional Lobule',                27: 'Subcallosal Cortex',
    28: 'Paracingulate Gyrus',                   29: 'Cingulate Gyrus (ant)',
    30: 'Cingulate Gyrus (post)',                31: 'Precuneous Cortex',
    32: 'Cuneal Cortex',                         33: 'Frontal Orbital Cortex',
    34: 'Parahippocampal Gyrus (ant)',           35: 'Parahippocampal Gyrus (post)',
    36: 'Lingual Gyrus',                         37: 'Temporal Fusiform (ant)',
    38: 'Temporal Fusiform (post)',              39: 'Temporal Occipital Fusiform',
    40: 'Occipital Fusiform Gyrus',              41: 'Frontal Operculum',
    42: 'Central Opercular Cortex',              43: 'Parietal Operculum',
    44: 'Planum Polare',                         45: "Heschl's Gyrus",
    46: 'Planum Temporale',                      47: 'Supracalcarine Cortex',
    48: 'Occipital Pole',                        49: 'L Thalamus',
    50: 'L Caudate',                             51: 'L Putamen',
    52: 'L Pallidum',                            53: 'Brain-Stem',
    54: 'L Hippocampus',                         55: 'L Amygdala',
    56: 'L Accumbens',                           57: 'R Thalamus',
    58: 'R Caudate',                             59: 'R Putamen',
    60: 'R Pallidum',                            61: 'R Hippocampus',
    62: 'R Amygdala',                            63: 'R Accumbens',
    64: 'Frontal Pole (R)',                      65: 'Insular Cortex (R)',
    66: 'Sup. Frontal Gyrus (R)',                67: 'Mid. Frontal Gyrus (R)',
    68: 'Inf. Frontal Gyrus tri (R)',            69: 'Inf. Frontal Gyrus oper (R)',
    70: 'Precentral Gyrus (R)',                  71: 'Temporal Pole (R)',
    72: 'Sup. Temporal Gyrus ant (R)',           73: 'Sup. Temporal Gyrus post (R)',
    74: 'Mid. Temporal Gyrus ant (R)',           75: 'Mid. Temporal Gyrus post (R)',
    76: 'Mid. Temporal Gyrus tO (R)',            77: 'Inf. Temporal Gyrus ant (R)',
    78: 'Inf. Temporal Gyrus post (R)',          79: 'Inf. Temporal Gyrus tO (R)',
    80: 'Postcentral Gyrus (R)',                 81: 'Sup. Parietal Lobule (R)',
    82: 'Supramarginal Gyrus ant (R)',           83: 'Supramarginal Gyrus post (R)',
    84: 'Angular Gyrus (R)',                     85: 'Lateral Occipital sup (R)',
    86: 'Lateral Occipital inf (R)',             87: 'Intracalcarine Cortex (R)',
    88: 'Frontal Medial Cortex (R)',             89: 'Juxtapositional Lobule (R)',
    90: 'Subcallosal Cortex (R)',                91: 'Paracingulate Gyrus (R)',
    92: 'Cingulate Gyrus ant (R)',               93: 'Cingulate Gyrus post (R)',
    94: 'Precuneous Cortex (R)',                 95: 'Cuneal Cortex (R)',
    96: 'Frontal Orbital Cortex (R)',            97: 'Parahippocampal Gyrus ant (R)',
    98: 'Parahippocampal Gyrus post (R)',        99: 'Lingual Gyrus (R)',
    100:'Temporal Fusiform ant (R)',             101:'Temporal Fusiform post (R)',
    102:'Temporal Occipital Fusiform (R)',       103:'Occipital Fusiform (R)',
    104:'Frontal Operculum (R)',                 105:'Central Opercular (R)',
    106:'Parietal Operculum (R)',                107:'Planum Polare (R)',
    108:"Heschl's Gyrus (R)",                   109:'Planum Temporale (R)',
    110:'Supracalcarine Cortex (R)',             111:'Occipital Pole (R)',
}

LOBE_MAP = {
    'Frontal':     [2,3,4,5,6,7,25,27,28,29,33,41,64,66,67,68,69,70,88,90,91,96,104],
    'Temporal':    [8,9,10,11,12,13,14,15,16,37,38,39,44,45,46,71,72,73,74,75,76,77,78,79,100,101,102,107,108,109],
    'Parietal':    [17,18,19,20,21,43,80,81,82,83,84,105,106],
    'Occipital':   [22,23,24,32,36,40,47,48,85,86,87,95,99,103,110,111],
    'Cingulate':   [29,30,31,92,93,94],
    'Subcortical': [34,35,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,97,98],
}

LOBE_COLORS = {
    'Frontal':     "#005EC1",
    'Temporal':    "#C64007",
    'Parietal':    "#0B9650",
    'Occipital':   "#A98100",
    'Cingulate':   "#3F0092",
    'Subcortical': "#920033",
    'Other':       "#311D1D",
}

def get_lobe(roi):
    for lobe, rois in LOBE_MAP.items():
        if roi in rois:
            return lobe
    return 'Other'

def get_label(roi):
    """Full node label: ROI index + region name."""
    name = HO_LABELS.get(roi, 'Unknown')
    return f"ROI{roi}\n{name}"

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
    Nodes: id, label, lobe, lobe_color, degree, total_importance
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
        lobe = get_lobe(roi)
        nodes.append({
            'id':               str(roi),
            'label':            get_label(roi),
            'lobe':             lobe,
            'lobe_color':       LOBE_COLORS[lobe],
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

    # Node color: passthrough from lobe_color column
    p4c.set_node_color_mapping(
        table_column='lobe_color',
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
    # Engine args -- only needed if running the C++ pipeline before graphing
    parser.add_argument('--run-engine',      action='store_true',
                        help='Run the C++ connectome engine before graphing')
    parser.add_argument('--asd-path',        type=str, default='',
                        help='Path to ASD .1D files (required if --run-engine)')
    parser.add_argument('--con-path',        type=str, default='',
                        help='Path to Control .1D files (required if --run-engine)')
    parser.add_argument('--ranger-path',     type=str, default='',
                        help='Path to Ranger executable (required if --run-engine)')
    parser.add_argument('--ntree',           type=int, default=1000,
                        help='Number of trees for Ranger (default: 1000)')
    parser.add_argument('--nthreads',        type=int, default=4,
                        help='Number of threads for Ranger (default: 4)')
    args = parser.parse_args()

    # Optionally run the C++ engine first to generate the importance file
    if args.run_engine:
        if not ENGINE_AVAILABLE:
            print("Error: connectome_engine module not built. See CMakeLists.txt.")
            return
        if not args.asd_path or not args.con_path or not args.ranger_path:
            print("Error: --asd-path, --con-path, and --ranger-path are required with --run-engine.")
            return
        print("Running C++ connectome engine...")
        result = connectome_engine.run(
            asd_path    = args.asd_path,
            con_path    = args.con_path,
            ranger_path = args.ranger_path,
            ntree       = args.ntree,
            nthreads    = args.nthreads,
            verbose     = True,
        )
        print(f"  Engine finished: {result['subjects_written']} subjects, status={result['ranger_status']}")
        if not result['success']:
            print("Error: Ranger failed. Check output above.")
            return

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
    print("\nTop 5 hub regions by aggregated importance:")
    for _, row in top_hubs.iterrows():
        roi = int(row['id'])
        print(f"  ROI{roi:>3} | {row['lobe']:>11} | degree={row['degree']} "
              f"| score={row['total_importance']:.4f} | {HO_LABELS.get(roi, 'Unknown')}")


if __name__ == '__main__':
    main()