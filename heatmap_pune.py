import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
from geopy.distance import geodesic

# Set random seed for reproducibility
np.random.seed(42)
num_locations = 100

# Generate random locations within Pune's geographical bounds
latitudes = np.random.uniform(18.45, 18.65, num_locations)
longitudes = np.random.uniform(73.75, 74.05, num_locations)

# Generate data for population density, orders, profit, and revenue
population_density = np.random.randint(500, 5000, num_locations)
orders = np.random.randint(50, 1000, num_locations)
profit = orders * np.random.uniform(50, 200, num_locations)
revenue = profit * np.random.uniform(1.2, 1.5, num_locations)

# Store data in a DataFrame
data = pd.DataFrame({
    "Latitude": latitudes,
    "Longitude": longitudes,
    "Population_Density": population_density,
    "Orders": orders,
    "Profit": profit,
    "Revenue": revenue
})

# Compute weighted score considering multiple factors
data["Weighted_Score"] = (
    data["Population_Density"] * 0.3 +
    data["Orders"] * 0.3 +
    data["Profit"] * 0.2 +
    data["Revenue"] * 0.2
)

# Normalize the weighted scores
data["Weighted_Score"] = (
    (data["Weighted_Score"] - data["Weighted_Score"].min()) /
    (data["Weighted_Score"].max() - data["Weighted_Score"].min())
)

# List of existing warehouse locations (example coordinates)
existing_warehouses = [(18.5204, 73.8567), (18.5310, 73.8745)]  # Add your existing warehouse locations here

# Function to check if a new location is too close to existing warehouses
def is_too_close(new_location, existing_locations, min_distance_km=5):
    for existing_location in existing_locations:
        if geodesic(new_location, existing_location).km < min_distance_km:
            return True
    return False

# Determine optimal warehouse locations using KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(data[["Latitude", "Longitude", "Weighted_Score"]])
warehouse_clusters = kmeans.cluster_centers_

# Filter out clusters that are too close to existing warehouses
filtered_clusters = []
for cluster in warehouse_clusters:
    latitude, longitude, _ = cluster
    if not is_too_close((latitude, longitude), existing_warehouses):
        filtered_clusters.append(cluster)

# Create a map of Pune
pune_map = folium.Map(location=[18.5204, 73.8567], zoom_start=12)

# Prepare heatmap data
heat_data = data[["Latitude", "Longitude", "Weighted_Score"]].values.tolist()
HeatMap(heat_data, radius=15, blur=10).add_to(pune_map)

# Add warehouse locations to the map
for i, cluster in enumerate(filtered_clusters):
    latitude, longitude, _ = cluster
    folium.Marker(
        location=[latitude, longitude],
        popup=(
            f"Warehouse Location {i+1}\n"
            f"Latitude: {latitude}\n"
            f"Longitude: {longitude}\n"
            "Reasoning: This location was selected based on the highest weighted score, considering population density, orders, profit, and revenue."
        ),
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(pune_map)

# Save the map
heatmap_file_path = "pune_warehouse_map.html"
pune_map.save(heatmap_file_path)
