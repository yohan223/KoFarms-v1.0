import streamlit as st
import geopandas as gpd
import folium
import ee
import pandas as pd
import numpy as np
import rasterio
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier

# Authenticate and Initialize Google Earth Engine
try:
    ee.Initialize(project='ee-johnslick1999')
except Exception as e:
    st.error(f"Failed to initialize Earth Engine: {e}")

# Streamlit app title
st.title("🌾 Smart Crop Recommendation System")

# Upload KML file
uploaded_file = st.file_uploader("Upload your KML file", type=["kml"])

if uploaded_file:
    try:
        # Read the KML file
        gdf = gpd.read_file(uploaded_file, driver="KML")
        st.success("KML uploaded successfully!")
        
        # Ensure the geometry is a polygon
        if gdf.geometry[0].geom_type != "Polygon":
            st.error("Uploaded KML does not contain a valid polygon.")
        else:
            roi = ee.Geometry.Polygon(list(gdf.geometry[0].exterior.coords))
    except Exception as e:
        st.error(f"Error processing KML file: {e}")
        roi = None

    if roi:
        # Function to get NDVI from Sentinel-2
        def get_ndvi():
            s2 = ee.ImageCollection("COPERNICUS/S2_SR")\
                .filterBounds(roi)\
                .filterDate("2024-01-01", "2024-12-31")\
                .median()
            ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
            return ndvi.clip(roi)
        
        # Function to get Soil Moisture from SMAP
        def get_soil_moisture():
            smap = ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture")\
                .filterBounds(roi)\
                .filterDate("2024-01-01", "2024-12-31")\
                .mean()
            return smap.select("ssm").clip(roi)
        
        # Function to get Heat Index from Sentinel-3
        def get_heat_index():
            lst = ee.ImageCollection("COPERNICUS/S3")\
                .filterBounds(roi)\
                .filterDate("2024-01-01", "2024-12-31")\
                .median()
            return lst.select("LST_averaged").clip(roi)

        # Extract data with loading indicators
        with st.spinner("Fetching NDVI data..."):
            try:
                ndvi = get_ndvi()
            except Exception as e:
                st.error(f"Error fetching NDVI: {e}")

        with st.spinner("Fetching Soil Moisture data..."):
            try:
                soil_moisture = get_soil_moisture()
            except Exception as e:
                st.error(f"Error fetching Soil Moisture: {e}")

        with st.spinner("Fetching Heat Index data..."):
            try:
                heat_index = get_heat_index()
            except Exception as e:
                st.error(f"Error fetching Heat Index: {e}")
        
        # Convert to Pandas DataFrame (Mock Data for Training ML Model)
        data = {
            "NDVI": np.random.uniform(0.2, 0.9, 100),
            "Soil_Moisture": np.random.uniform(0.1, 0.5, 100),
            "Heat_Index": np.random.uniform(25, 40, 100),
            "Crop": np.random.choice(["Maize", "Wheat", "Rice", "Soybeans"], 100)
        }
        df = pd.DataFrame(data)
        
        # Train a basic Random Forest Model
        X = df[["NDVI", "Soil_Moisture", "Heat_Index"]]
        y = df["Crop"]
        model = RandomForestClassifier()
        model.fit(X, y)
        
        # Predict best crop for the uploaded field
        with st.spinner("Processing crop recommendation..."):
            try:
                field_data = pd.DataFrame({
                    "NDVI": [ndvi.reduceRegion(ee.Reducer.mean(), roi, 100).get("NDVI").getInfo()],
                    "Soil_Moisture": [soil_moisture.reduceRegion(ee.Reducer.mean(), roi, 100).get("ssm").getInfo()],
                    "Heat_Index": [heat_index.reduceRegion(ee.Reducer.mean(), roi, 100).get("LST_averaged").getInfo()]
                })
                best_crop = model.predict(field_data)[0]
                st.subheader("🌿 Best Crop Recommendation:")
                st.success(f"The best crop to plant is: **{best_crop}**")
            except Exception as e:
                st.error(f"Error in crop recommendation: {e}")
        
        # Display map
        st.subheader("🗺️ Field Location:")
        m = folium.Map(location=[gdf.geometry[0].centroid.y, gdf.geometry[0].centroid.x], zoom_start=14)
        folium.GeoJson(gdf).add_to(m)
        st_folium(m, width=700, height=500)

st.write("✅ KML file processed!")
