import subprocess
import sys
import os

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
