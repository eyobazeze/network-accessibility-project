import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PLACE_NAME = "Addis Ababa, Ethiopia"
CRS_PROJECTED = "EPSG:32637"

# Pull the real street network
print("STEP 1: Fetching street network from OpenStreetMap...")
G = ox.graph_from_place(PLACE_NAME, network_type="walk")
print(f"  ✓ Network loaded: {len(G.nodes):,} nodes, {len(G.edges):,} edges")