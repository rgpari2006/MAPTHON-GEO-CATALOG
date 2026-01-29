import shapefile
import os

# Create Data/vector directory if it doesn't exist
os.makedirs('Data/vector', exist_ok=True)

# Create India Country Boundary Shapefile
print("Creating India_Country_Boundary.shp...")
w = shapefile.Writer('Data/vector/India_Country_Boundary')
w.field('name', 'C')
w.field('area_sqkm', 'N', decimal=2)

# Define India's approximate bounding box as a polygon (simplified)
# Coordinates: [longitude, latitude]
india_bounds = [
    [68.7, 8.4],    # Southwest corner
    [97.4, 8.4],    # Southeast corner
    [97.4, 35.5],   # Northeast corner
    [68.7, 35.5],   # Northwest corner
    [68.7, 8.4]     # Close the polygon
]

w.poly([india_bounds])
w.record('India', 3287263.0)
w.close()

# Create .prj file for India_Country_Boundary (WGS84)
with open('Data/vector/India_Country_Boundary.prj', 'w') as f:
    f.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')

print("✓ India_Country_Boundary.shp created successfully")

# Create India State Boundary Shapefile (sample: two states)
print("Creating India_State_Boundary.shp...")
w = shapefile.Writer('Data/vector/India_State_Boundary')
w.field('state_name', 'C')
w.field('area_sqkm', 'N', decimal=2)

# Maharashtra state boundary (simplified)
maharashtra = [
    [72.6, 15.6],
    [80.9, 15.6],
    [80.9, 22.0],
    [72.6, 22.0],
    [72.6, 15.6]
]

# Karnataka state boundary (simplified)
karnataka = [
    [74.1, 11.5],
    [78.6, 11.5],
    [78.6, 18.5],
    [74.1, 18.5],
    [74.1, 11.5]
]

w.poly([maharashtra])
w.record('Maharashtra', 307713.0)

w.poly([karnataka])
w.record('Karnataka', 191791.0)

w.close()

# Create .prj file for India_State_Boundary (WGS84)
with open('Data/vector/India_State_Boundary.prj', 'w') as f:
    f.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')

print("✓ India_State_Boundary.shp created successfully")

# List all created files
print("\nCreated files:")
for file in sorted(os.listdir('Data/vector')):
    filepath = os.path.join('Data/vector', file)
    size = os.path.getsize(filepath)
    print(f"  {file} ({size} bytes)")
