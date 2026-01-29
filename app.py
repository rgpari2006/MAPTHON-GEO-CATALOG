import streamlit as st
import pandas as pd
import os
from pathlib import Path
import rasterio
from rasterio.errors import RasterioIOError
import geopandas as gpd
import fiona
from fiona.errors import FionaValueError
import traceback
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
import io
import subprocess
import sys

# Set page config
st.set_page_config(
    page_title="Mapathon Geo-Catalog",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Mapathon Geo-Catalog")
st.markdown("**Discover and explore geospatial data with automated metadata extraction**")

# Initialize session state for catalog data
if 'catalog_df' not in st.session_state:
    st.session_state.catalog_df = None
if 'scan_completed' not in st.session_state:
    st.session_state.scan_completed = False


def calculate_quality_score(metadata: Dict[str, Any], file_type: str) -> int:
    """
    Calculate quality score (0-100) for a geospatial file.
    
    Deductions:
    - 10 points if CRS is missing
    - 5 points if nodata is missing (for rasters)
    """
    score = 100
    
    # Check CRS
    if metadata.get('crs') is None or metadata.get('crs') == '':
        score -= 10
    
    # Check nodata (for rasters only)
    if file_type == 'GeoTIFF' and metadata.get('nodata') is None:
        score -= 5
    
    return max(0, score)  # Ensure score doesn't go below 0


def extract_geotiff_metadata(filepath: str) -> Dict[str, Any]:
    """
    Extract metadata from GeoTIFF files using rasterio.
    """
    try:
        with rasterio.open(filepath) as src:
            metadata = {
                'filename': os.path.basename(filepath),
                'filepath': filepath,
                'file_type': 'GeoTIFF',
                'driver': src.driver,
                'width': src.width,
                'height': src.height,
                'band_count': src.count,
                'crs': str(src.crs) if src.crs else None,
                'dtype': src.dtypes[0] if src.dtypes else None,
                'nodata': src.nodata,
                'bounds': str(src.bounds) if hasattr(src, 'bounds') else None,
                'resolution': str(src.res) if hasattr(src, 'res') else None,
            }
            metadata['quality_score'] = calculate_quality_score(metadata, 'GeoTIFF')
            return metadata
    except (RasterioIOError, Exception) as e:
        st.warning(f"⚠️ Error reading GeoTIFF {os.path.basename(filepath)}: {str(e)}")
        return None


def extract_shapefile_metadata(filepath: str) -> Dict[str, Any]:
    """
    Extract metadata from Shapefile using geopandas/fiona.
    """
    try:
        # Check if all required shapefile components exist
        base_path = filepath.rsplit('.', 1)[0]
        required_extensions = ['.shp', '.shx', '.dbf']
        missing_files = []
        
        for ext in required_extensions:
            if not os.path.exists(base_path + ext):
                missing_files.append(ext)
        
        if missing_files:
            st.warning(f"⚠️ Shapefile {os.path.basename(filepath)} is incomplete. Missing files: {', '.join(missing_files)}")
            return None
        
        # Use fiona for basic metadata
        with fiona.open(filepath) as src:
            crs = src.crs
            geometry_type = src.schema.get('geometry', 'Unknown')
            feature_count = len(src)
        
        metadata = {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'file_type': 'Shapefile',
            'crs': str(crs) if crs else None,
            'geometry_type': geometry_type,
            'feature_count': feature_count,
        }
        metadata['quality_score'] = calculate_quality_score(metadata, 'Shapefile')
        return metadata
    except (FionaValueError, Exception) as e:
        st.warning(f"⚠️ Error reading Shapefile {os.path.basename(filepath)}: {str(e)}")
        return None


def convert_raster_to_image(filepath: str) -> tuple:
    """
    Convert raster file to PIL Image for web display.
    Returns tuple of (PIL Image, metadata)
    """
    try:
        with rasterio.open(filepath) as src:
            metadata = {
                'width': src.width,
                'height': src.height,
                'bands': src.count,
                'crs': str(src.crs) if src.crs else 'Not defined',
                'dtype': src.dtypes[0] if src.dtypes else 'Unknown',
                'nodata': src.nodata if src.nodata is not None else 'None',
                'bounds': str(src.bounds)
            }
            
            # Read the first band for single-band display
            if src.count >= 3:
                # For multi-band, read RGB bands
                red = src.read(1)
                green = src.read(2)
                blue = src.read(3)
                
                # Normalize values to 0-255
                red_norm = ((red - red.min()) / (red.max() - red.min()) * 255).astype(np.uint8)
                green_norm = ((green - green.min()) / (green.max() - green.min()) * 255).astype(np.uint8)
                blue_norm = ((blue - blue.min()) / (blue.max() - blue.min()) * 255).astype(np.uint8)
                
                # Stack bands
                image_array = np.stack([red_norm, green_norm, blue_norm], axis=-1)
            else:
                # For single or two-band, display first band as grayscale
                image = src.read(1)
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                # Convert grayscale to RGB for PIL
                image_array = np.stack([image, image, image], axis=-1)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_array.astype(np.uint8), 'RGB')
            return pil_image, metadata
    except Exception as e:
        st.error(f"❌ Error converting raster: {str(e)}")
        return None, None


def display_raster_image(filepath: str):
    """
    Display raster image using streamlit's st.image (optimized for web).
    """
    try:
        pil_image, metadata = convert_raster_to_image(filepath)
        
        if pil_image and metadata:
            # Display image as web-friendly format
            st.image(pil_image, caption=f"Raster Image: {os.path.basename(filepath)}", use_container_width=True)
            
            # Display metadata
            st.subheader("📋 Raster Details")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Width:** {metadata['width']} pixels")
            with col2:
                st.info(f"**Height:** {metadata['height']} pixels")
            with col3:
                st.info(f"**Bands:** {metadata['bands']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**CRS:** {metadata['crs']}")
            with col2:
                st.info(f"**Data Type:** {metadata['dtype']}")
            with col3:
                st.info(f"**NoData Value:** {metadata['nodata']}")
            
            st.info(f"**Bounds:** {metadata['bounds']}")
            
    except Exception as e:
        st.error(f"❌ Error displaying raster image: {str(e)}")
        st.error(traceback.format_exc())


def scan_directory(target_dir: str) -> pd.DataFrame:
    """
    Recursively scan directory for geospatial files and extract metadata.
    """
    geospatial_files = []
    supported_extensions = {'.tif', '.tiff', '.shp'}
    
    if not os.path.exists(target_dir):
        st.error(f"❌ Directory not found: {target_dir}")
        return pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Collect all files first
    all_files = []
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in supported_extensions:
                all_files.append(os.path.join(root, file))
    
    total_files = len(all_files)
    
    if total_files == 0:
        st.warning(f"⚠️ No geospatial files found in {target_dir}")
        return pd.DataFrame()
    
    # Process each file
    for idx, filepath in enumerate(all_files):
        ext = Path(filepath).suffix.lower()
        
        try:
            if ext in ['.tif', '.tiff']:
                metadata = extract_geotiff_metadata(filepath)
                if metadata:
                    geospatial_files.append(metadata)
            elif ext == '.shp':
                metadata = extract_shapefile_metadata(filepath)
                if metadata:
                    geospatial_files.append(metadata)
        except Exception as e:
            st.warning(f"⚠️ Unexpected error processing {os.path.basename(filepath)}: {str(e)}")
        
        # Update progress
        progress = (idx + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processed {idx + 1}/{total_files} files...")
    
    progress_bar.empty()
    status_text.empty()
    
    if not geospatial_files:
        st.warning("⚠️ No valid geospatial files could be read")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(geospatial_files)
    
    # Sort by quality score (descending)
    df = df.sort_values('quality_score', ascending=False).reset_index(drop=True)
    
    return df


# Sidebar controls
st.sidebar.header("🔍 Catalog Scanner")

target_dir = st.sidebar.text_input(
    "Target Directory",
    value="Data",
    help="Path to scan for geospatial files (relative or absolute)"
)

if st.sidebar.button("🔄 Scan Directory", key="scan_btn", use_container_width=True):
    with st.spinner("Scanning directory and extracting metadata..."):
        full_path = target_dir if os.path.isabs(target_dir) else os.path.join(os.getcwd(), target_dir)
        st.session_state.catalog_df = scan_directory(full_path)
        st.session_state.scan_completed = True
    
    if not st.session_state.catalog_df.empty:
        st.success(f"✅ Found {len(st.session_state.catalog_df)} geospatial files!")
    else:
        st.error("❌ No files found or an error occurred during scan.")

# Display catalog if data exists
if st.session_state.scan_completed and st.session_state.catalog_df is not None and not st.session_state.catalog_df.empty:
    st.markdown("---")
    st.header("📊 Catalog & Filters")
    
    df = st.session_state.catalog_df.copy()
    
    # Create filter columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        file_types = ['All'] + sorted(df['file_type'].unique().tolist())
        selected_file_type = st.selectbox(
            "File Type",
            options=file_types,
            key="file_type_filter"
        )
    
    with col2:
        crs_options = ['All'] + sorted(df['crs'].dropna().unique().tolist())
        selected_crs = st.selectbox(
            "CRS",
            options=crs_options,
            key="crs_filter"
        )
    
    with col3:
        # Only show dtype filter for rasters
        raster_df = df[df['file_type'] == 'GeoTIFF']
        if not raster_df.empty:
            dtype_options = ['All'] + sorted(raster_df['dtype'].dropna().unique().tolist())
            selected_dtype = st.selectbox(
                "Data Type (Raster)",
                options=dtype_options,
                key="dtype_filter"
            )
        else:
            selected_dtype = 'All'
    
    with col4:
        quality_score_range = st.slider(
            "Minimum Quality Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="quality_filter"
        )
    
    # Apply filters
    if selected_file_type != 'All':
        df = df[df['file_type'] == selected_file_type]
    
    if selected_crs != 'All':
        df = df[df['crs'] == selected_crs]
    
    if selected_dtype != 'All' and 'dtype' in df.columns:
        df = df[df['dtype'] == selected_dtype]
    
    df = df[df['quality_score'] >= quality_score_range]
    
    # Display summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files", len(df))
    with col2:
        st.metric("Avg Quality Score", f"{df['quality_score'].mean():.1f}")
    with col3:
        st.metric("File Types", df['file_type'].nunique())
    with col4:
        st.metric("CRS Count", df['crs'].nunique())
    
    st.markdown("---")
    
    # Display table
    st.subheader("📑 File Catalog")
    
    # Prepare display columns based on file types present
    display_df = df.copy()
    
    # Reorder columns for better visibility
    column_order = ['filename', 'file_type', 'quality_score']
    
    # Add file-type-specific columns
    if 'GeoTIFF' in display_df['file_type'].values:
        column_order.extend(['width', 'height', 'band_count', 'crs', 'dtype', 'nodata'])
    if 'Shapefile' in display_df['file_type'].values:
        if 'geometry_type' not in column_order:
            column_order.extend(['geometry_type', 'feature_count'])
        # Don't add 'crs' again if it already exists
        if 'crs' not in column_order and 'crs' in display_df.columns:
            column_order.append('crs')
    
    # Keep only available columns and remove duplicates while preserving order
    seen = set()
    column_order = [col for col in column_order if col in display_df.columns and not (col in seen or seen.add(col))]
    display_df = display_df[column_order]
    
    # Display with Streamlit
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "quality_score": st.column_config.NumberColumn(
                "Quality Score",
                format="%d ⭐"
            ),
            "filename": st.column_config.TextColumn(
                "Filename",
                width="medium"
            ),
            "file_type": st.column_config.TextColumn(
                "File Type",
                width="small"
            ),
        }
    )
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Catalog as CSV",
        data=csv,
        file_name="geospatial_catalog.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Raster Image Viewer
    st.markdown("---")
    st.header("🖼️ Raster Image Viewer")
    
    # Get raster files only
    raster_files = df[df['file_type'] == 'GeoTIFF'].copy()
    
    if not raster_files.empty:
        # Create tabs for different view modes
        tab1, tab2 = st.tabs(["🔍 Detailed View", "📑 Quick Preview Grid"])
        
        with tab1:
            selected_raster = st.selectbox(
                "Select a raster file to view:",
                options=raster_files['filename'].tolist(),
                key="raster_viewer"
            )
            
            # Get the filepath of selected raster
            if selected_raster:
                raster_filepath = raster_files[raster_files['filename'] == selected_raster]['filepath'].values[0]
                
                # Display the raster image
                display_raster_image(raster_filepath)
        
        with tab2:
            st.subheader("TIF File Previews")
            # Display all TIF images in a grid
            cols = st.columns(2)
            for idx, (_, row) in enumerate(raster_files.iterrows()):
                col = cols[idx % 2]
                with col:
                    st.write(f"**{row['filename']}**")
                    try:
                        pil_image, metadata = convert_raster_to_image(row['filepath'])
                        if pil_image:
                            st.image(pil_image, use_container_width=True)
                            st.caption(f"Size: {metadata['width']}×{metadata['height']} | Bands: {metadata['bands']} | CRS: {metadata['crs']}")
                    except Exception as e:
                        st.error(f"Error loading {row['filename']}: {str(e)}")
    else:
        st.info("📌 No raster files found in the current catalog. Try adjusting filters or scanning a different directory.")

elif st.session_state.scan_completed and (st.session_state.catalog_df is None or st.session_state.catalog_df.empty):
    st.info("👈 Click the **'Scan Directory'** button in the sidebar to discover geospatial files.")
else:
    st.info("👈 Click the **'Scan Directory'** button in the sidebar to discovergeospatial files.")




def main():
    """Launch the Mapathon Geo-Catalog Streamlit app."""
    print("🗺 Starting Mapathon Geo-Catalog...")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("✓ Created 'data/' directory")
    
    # Run streamlit
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            check=True
        )
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Install with: pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✓ Application closed.")
        sys.exit(0)

if __name__ == "__main__":
    main()
