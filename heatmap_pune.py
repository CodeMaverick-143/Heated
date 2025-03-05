import numpy as np
import pandas as pd
import json
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans

np.random.seed(42)
num_locations = 100
latitudes = np.random.uniform(18.45, 18.65, num_locations)
longitudes = np.random.uniform(73.75, 74.05, num_locations)
population_density = np.random.randint(500, 5000, num_locations)
orders = np.random.randint(50, 1000, num_locations)
profit = orders * np.random.uniform(50, 200, num_locations)
revenue = profit * np.random.uniform(1.2, 1.5, num_locations)

data = pd.DataFrame({
    "Latitude": latitudes,
    "Longitude": longitudes,
    "Population_Density": population_density,
    "Orders": orders,
    "Profit": profit,
    "Revenue": revenue
})

# Generate more random data for warehouse locations
warehouse_latitudes = np.random.uniform(18.45, 18.65, 5)
warehouse_longitudes = np.random.uniform(73.75, 74.05, 5)
warehouse_locations = pd.DataFrame({
    "Latitude": warehouse_latitudes,
    "Longitude": warehouse_longitudes
})

# Assign weighted score to each point based on multiple factors
data["Weighted_Score"] = (
    data["Population_Density"] * 0.3 +
    data["Orders"] * 0.3 +
    data["Profit"] * 0.2 +
    data["Revenue"] * 0.2
)

# Normalize the weighted scores
data["Weighted_Score"] = (data["Weighted_Score"] - data["Weighted_Score"].min()) / (data["Weighted_Score"].max() - data["Weighted_Score"].min())

# Use KMeans clustering to determine the optimal warehouse locations
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data[["Latitude", "Longitude", "Weighted_Score"]])
warehouse_clusters = kmeans.cluster_centers_

# Create the map of Pune
pune_map = folium.Map(location=[18.5204, 73.8567], zoom_start=12)

# Prepare heatmap data
heat_data = data[["Latitude", "Longitude", "Weighted_Score"]].values.tolist()
HeatMap(heat_data, radius=15, blur=10).add_to(pune_map)

# Add a legend for intensity scale
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 90px; background-color: white; z-index:9999; font-size:14px; padding:10px; border-radius:5px;">
    <b>Heatmap Intensity</b><br>
    <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> High<br>
    <i style="background: yellow; width: 10px; height: 10px; display: inline-block;"></i> Medium<br>
    <i style="background: green; width: 10px; height: 10px; display: inline-block;"></i> Low<br>
    <b>Ranges:</b><br>
    Population Density: 500-5000<br>
    Orders: 50-1000<br>
    Profit: 2500-200000<br>
    Revenue: 3000-300000
</div>
'''
pune_map.get_root().html.add_child(folium.Element(legend_html))

# Add markers for the optimal warehouse locations (cluster centroids)
for i, cluster in enumerate(warehouse_clusters):
    latitude, longitude, _ = cluster
    folium.Marker(
        location=[latitude, longitude],
        popup=(
            f"Warehouse Location {i+1}\n"
            f"Latitude: {latitude}\n"
            f"Longitude: {longitude}\n"
            f"Reasoning: This location was selected based on the highest weighted score (Population Density, Orders, Profit, Revenue)."
        ),
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(pune_map)

# Save the map with the heatmap and warehouse locations
heatmap_file_path = "pune_warehouse_map.html"
pune_map.save(heatmap_file_path)
