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

json_file_path = "JSON.json"
with open(json_file_path, "w") as json_file:
    json.dump(data.to_dict(orient="records"), json_file, indent=4)

features = data[["Latitude", "Longitude", "Population_Density", "Orders", "Profit", "Revenue"]]
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(features)
warehouse_locations = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)[["Latitude", "Longitude"]]

warehouse_json_file_path = "JSON_2.json"
with open(warehouse_json_file_path, "w") as json_file:
    json.dump(warehouse_locations.to_dict(orient="records"), json_file, indent=4)

pune_map = folium.Map(location=[18.5204, 73.8567], zoom_start=12)

data["Weighted_Score"] = (
    data["Population_Density"] * 0.3 +
    data["Orders"] * 0.3 +
    data["Profit"] * 0.2 +
    data["Revenue"] * 0.2
)
data["Weighted_Score"] = (data["Weighted_Score"] - data["Weighted_Score"].min()) / (data["Weighted_Score"].max() - data["Weighted_Score"].min())

heat_data = data[["Latitude", "Longitude", "Weighted_Score"]].values.tolist()
HeatMap(heat_data, radius=15, blur=10).add_to(pune_map)

# Add a legend for intensity scale
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 90px; background-color: white; z-index:9999; font-size:14px; padding:10px; border-radius:5px;">
    <b>Heatmap Intensity</b><br>
    <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> High<br>
    <i style="background: yellow; width: 10px; height: 10px; display: inline-block;"></i> Medium<br>
    <i style="background: green; width: 10px; height: 10px; display: inline-block;"></i> Low
</div>
'''
pune_map.get_root().html.add_child(folium.Element(legend_html))

for _, row in warehouse_locations.iterrows():
    existing_warehouse = data[(data["Latitude"] == row["Latitude"]) & (data["Longitude"] == row["Longitude"])]
    reasoning = "This location is chosen due to high population density and order volume."
    if not existing_warehouse.empty:
        reasoning += " However, there is already a warehouse here, which may require optimization."
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=(
            f"Warehouse Location\n"
            f"Reason for selection: {reasoning}\n"
            f"Why not other sites? Lower density or profit\n"
            f"Existing warehouse impact? {reasoning}"
        ),
        icon=folium.Icon(color="red", icon="home")
    ).add_to(pune_map)

heatmap_file_path = "index.html"
pune_map.save(heatmap_file_path)
