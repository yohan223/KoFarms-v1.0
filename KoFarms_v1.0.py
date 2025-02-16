
# Ensure authentication
ee.Authenticate()
##try:
   ##ee.Initialize()
#except Exception as e:
    #ee.Authenticate()
    #ee.Initialize()##
    
# Load datasets
smap = ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture")
soil_temp = ee.ImageCollection("OpenLandMap/SOL/SOILTEMP")
soil_texture = ee.ImageCollection("OpenLandMap/SOL/SOILTEXTURE")
ndvi = ee.ImageCollection("COPERNICUS/S2").filterDate(str(start_date), str(end_date)).filterBounds(roi)

# Compute mean values
smap_mean = smap.filterDate(str(start_date), str(end_date)).mean().clip(roi)
soil_temp_mean = soil_temp.mean().clip(roi)
soil_texture_mean = soil_texture.mean().clip(roi)
ndvi_mean = ndvi.mean().clip(roi)

# Extract values as dictionaries
smap_data = smap_mean.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=1000).getInfo()
soil_temp_data = soil_temp_mean.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=1000).getInfo()
soil_texture_data = soil_texture_mean.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=1000).getInfo()
ndvi_data = ndvi_mean.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=10).getInfo()

# Create DataFrame
results = {"Parameter": ["Soil Moisture", "Soil Temperature", "Soil Texture", "NDVI"],
           "Value": [smap_data, soil_temp_data, soil_texture_data, ndvi_data]}
df = pd.DataFrame(results)

# Display results
st.write("### Analysis Results")
st.dataframe(df)

# Save CSV
csv_file = "crop_analysis_results.csv"
df.to_csv(csv_file, index=False)
st.download_button(label="Download CSV", data=df.to_csv().encode(), file_name=csv_file, mime="text/csv")

# Export clipped satellite image as TIFF
def export_image(image, filename):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=filename,
        scale=30,
        region=roi.getInfo(),
        fileFormat='GeoTIFF'
    )
    task.start()
    return task.status()

# Export NDVI image
export_status = export_image(ndvi_mean, "NDVI_Clipped")
st.write("Export Status:", export_status)
