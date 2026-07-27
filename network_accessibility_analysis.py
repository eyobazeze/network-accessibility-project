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


# Get destination points (hospitals & schools)
print("STEP 2: Fetching hospitals and schools from OpenStreetMap...")

tags_healthcare = {"amenity": "hospital"}
tags_schools = {"amenity": "school"}

hospitals = ox.features_from_place(PLACE_NAME, tags_healthcare)
schools = ox.features_from_place(PLACE_NAME, tags_schools)

print(f"  ✓ Hospitals found: {len(hospitals)}")
print(f"  ✓ Schools found: {len(schools)}")

# Keep only point geometries with valid coordinates (some OSM features are
# polygons/areas rather than points — convert to centroid for consistency)
hospitals = hospitals[hospitals.geometry.notna()].copy()
hospitals["geometry"] = hospitals.geometry.centroid

schools = schools[schools.geometry.notna()].copy()
schools["geometry"] = schools.geometry.centroid


# Match origins and destinations to the street network
print("STEP 3: Matching points to the street network...")

# Reproject graph to meters for accurate distance calculations
G_proj = ox.project_graph(G, to_crs=CRS_PROJECTED)

# Reuse your population grid from the WorldPop project if available,
# otherwise fall back to a fresh grid — adjust path to your real file
pop_grid = gpd.read_file(DATA_DIR / "processed" / "subcities_with_real_population.geojson")
pop_grid = pop_grid.to_crs(CRS_PROJECTED)

# Get centroid coordinates for matching (works whether pop_grid holds points or polygons)
origin_coords = pop_grid.geometry.centroid
origin_x = origin_coords.x.values
origin_y = origin_coords.y.values

# Reproject destinations to match
hospitals_proj = hospitals.to_crs(CRS_PROJECTED)
schools_proj = schools.to_crs(CRS_PROJECTED)

dest_x = list(hospitals_proj.geometry.x) + list(schools_proj.geometry.x)
dest_y = list(hospitals_proj.geometry.y) + list(schools_proj.geometry.y)

# Find nearest network node to every origin and destination
origin_nodes = ox.distance.nearest_nodes(G_proj, X=origin_x, Y=origin_y)
dest_nodes = ox.distance.nearest_nodes(G_proj, X=dest_x, Y=dest_y)

print(f"  ✓ Matched {len(origin_nodes)} origins and {len(dest_nodes)} destinations to the network")


# Compute shortest network distance from each origin to its nearest destination
print("STEP 4: Computing network distances (this may take a few minutes)...")

# multi_source_dijkstra_path_length computes distance from ANY of the
# destination nodes simultaneously — much faster than looping origin-by-origin
# when you have many origins and relatively few destinations
distances_from_destinations = nx.multi_source_dijkstra_path_length(
    G_proj, sources=set(dest_nodes), weight="length"
)

# Look up each origin's distance to its nearest destination
network_distances = []
for node in origin_nodes:
    dist = distances_from_destinations.get(node, np.nan)  # NaN if unreachable
    network_distances.append(dist)

pop_grid["network_distance_m"] = network_distances

print(f"  ✓ Computed distances for {len(network_distances)} points")
print(f"  ✓ Unreachable points (network gaps): {pop_grid['network_distance_m'].isna().sum()}")


# Compare against straight-line distance (the actual finding)
print("STEP 5: Comparing network distance vs. straight-line distance...")

from shapely.geometry import Point

# Nearest straight-line distance for the same origins (for comparison)
all_dest_points = list(hospitals_proj.geometry) + list(schools_proj.geometry)

straight_line_distances = []
for origin_point in origin_coords:
    min_dist = min(origin_point.distance(d) for d in all_dest_points)
    straight_line_distances.append(min_dist)

pop_grid["straight_line_distance_m"] = straight_line_distances
pop_grid["network_penalty_ratio"] = pop_grid["network_distance_m"] / pop_grid["straight_line_distance_m"]

print(f"  ✓ Mean straight-line distance: {pop_grid['straight_line_distance_m'].mean():.0f}m")
print(f"  ✓ Mean network distance: {pop_grid['network_distance_m'].mean():.0f}m")
print(f"  ✓ Mean penalty ratio: {pop_grid['network_penalty_ratio'].mean():.2f}x")


# Map it
print("STEP 6: Generating comparison map...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

pop_grid.plot(column="straight_line_distance_m", cmap="YlOrRd", legend=True, ax=axes[0], markersize=15)
axes[0].set_title("Straight-Line Distance to Nearest Hospital/School (m)")

pop_grid.plot(column="network_distance_m", cmap="YlOrRd", legend=True, ax=axes[1], markersize=15)
axes[1].set_title("Real Network-Based Walking Distance (m)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "network_vs_straightline_comparison.png", dpi=150)
print(f"  ✓ Map saved → {OUTPUT_DIR / 'network_vs_straightline_comparison.png'}")

pop_grid.to_file(OUTPUT_DIR / "network_accessibility_results.geojson", driver="GeoJSON")
