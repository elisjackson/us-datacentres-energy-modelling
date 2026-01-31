import sqlite3
import geopandas as gpd
from shapely.ops import unary_union
from shapely.validation import make_valid

# -------------------------
# Configuration
# -------------------------
RAW_GPKG = r"C:\Users\Elis\repos\us-datacentres\data\downloads\shapefiles\geoBoundariesCGAZ_ADM0.gpkg"
NEW_GPKG = r"C:\Users\Elis\repos\us-datacentres\data\processed\geoBoundariesCGAZ_ADM0_processed.gpkg"

LAYER_IN = "globalADM0"
LAYER_UNION = "land_union"

INDEX_NAME = "idx_globalADM0_shapeName"
INDEX_COL = "shapeName"


def index_exists(conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 1
        FROM sqlite_master
        WHERE type = 'index'
          AND name = ?
    """, (INDEX_NAME,))
    return cursor.fetchone() is not None


def create_index(conn):
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS {INDEX_NAME}
        ON {LAYER_IN}({INDEX_COL});
    """)
    conn.commit()


def create_union_layer():
    # Read the raw layer
    gdf = gpd.read_file(RAW_GPKG, layer=LAYER_IN)
    
    print(f"Loaded {len(gdf)} geometries")
    
    # Validate and fix geometries before union
    print("Validating and fixing geometries...")
    invalid_count = 0
    empty_count = 0
    
    def fix_geometry(geom):
        """Fix a single geometry if invalid."""
        nonlocal invalid_count, empty_count
        
        if geom is None or geom.is_empty:
            empty_count += 1
            return None
        
        if not geom.is_valid:
            invalid_count += 1
            # Try to fix with make_valid
            try:
                fixed = make_valid(geom)
                # If make_valid returns a GeometryCollection, try to extract the largest polygon
                if hasattr(fixed, 'geoms'):
                    # Find the largest geometry in the collection
                    largest = max(fixed.geoms, key=lambda g: g.area if hasattr(g, 'area') else 0)
                    return largest if largest.is_valid else None
                return fixed if fixed.is_valid else None
            except Exception as e:
                print(f"Warning: Could not fix geometry: {e}")
                return None
        
        return geom
    
    # Fix all geometries
    gdf['geometry'] = gdf.geometry.apply(fix_geometry)
    
    # Remove None/empty geometries
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    
    print(f"Fixed {invalid_count} invalid geometries")
    print(f"Removed {empty_count} empty geometries")
    print(f"Processing union with {len(gdf)} valid geometries...")
    
    # Union all geometries with error handling
    try:
        land_union = unary_union(gdf.geometry)
    except Exception as e:
        print(f"Error during unary_union: {e}")
        print("Attempting alternative approach: buffering geometries to fix topology...")
        
        # Alternative: buffer(0) trick to fix topology issues
        # This can help with self-intersections and other topology problems
        gdf['geometry'] = gdf.geometry.buffer(0)
        
        # Re-validate after buffering
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid]
        print(f"After buffering: {len(gdf)} valid geometries")
        
        # Try union again
        try:
            land_union = unary_union(gdf.geometry)
        except Exception as e2:
            print(f"Error still occurs after buffering: {e2}")
            print("Attempting chunked union approach...")
            
            # Last resort: union in chunks
            chunks = []
            chunk_size = 100  # Process 100 geometries at a time
            
            num_chunks = (len(gdf) - 1) // chunk_size + 1
            for i in range(0, len(gdf), chunk_size):
                chunk = gdf.iloc[i:i+chunk_size]
                chunk_num = i // chunk_size + 1
                try:
                    chunk_union = unary_union(chunk.geometry)
                    if chunk_union is not None and not chunk_union.is_empty:
                        chunks.append(chunk_union)
                    print(f"Processed chunk {chunk_num}/{num_chunks}")
                except Exception as chunk_e:
                    print(f"Warning: Failed to union chunk {chunk_num}/{num_chunks}: {chunk_e}")
                    continue
            
            if chunks:
                land_union = unary_union(chunks)
            else:
                raise ValueError("Failed to create union: all chunks failed")

    # ---- VALIDATION CHECK ----
    if not land_union.is_valid:
        print("Union geometry is INVALID. Attempting to fix with make_valid()...")
        land_union = make_valid(land_union)

        if not land_union.is_valid:
            print("Warning: union geometry still invalid after make_valid().")
            raise ValueError("Union geometry still invalid after make_valid().")
        else:
            print("Union geometry successfully fixed.")

    # Create GeoDataFrame for union layer
    union_gdf = gpd.GeoDataFrame(geometry=[land_union], crs=gdf.crs)

    return union_gdf


def save_union_to_new_gpkg(union_gdf):
    # Copy the raw gpkg to a new one (preserves original)
    import shutil
    shutil.copy(RAW_GPKG, NEW_GPKG)

    # Save the union layer into the new gpkg
    union_gdf.to_file(NEW_GPKG, layer=LAYER_UNION, driver="GPKG")


def explore_union_layer():
    gdf = gpd.read_file(NEW_GPKG, layer=LAYER_UNION)

    # Ensure WGS84 for web mapping
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    m = gdf.explore(
        tooltip=True,
        popup=True,
        tiles="CartoDB positron"
    )

    out_html = r"C:\Users\Elis\repos\us-datacentres\data\processed\geoBoundariesCGAZ_ADM0_union_view.html"
    m.save(out_html)
    print(f"Saved interactive view to {out_html}")


def main():
    # 1) Check index, create if missing
    conn = sqlite3.connect(RAW_GPKG)
    if index_exists(conn):
        print("Index already exists.")
    else:
        print("Index not found. Creating index...")
        create_index(conn)
        print("Index created.")
    conn.close()

    # 2) Create union layer
    print("Creating union layer...")
    union_gdf = create_union_layer()
    print("Union created.")

    # 3) Save to new gpkg
    print("Saving to new GeoPackage...")
    save_union_to_new_gpkg(union_gdf)
    print(f"Saved new gpkg: {NEW_GPKG}")

    # 4) Explore union
    print("Creating interactive HTML preview...")
    explore_union_layer()


if __name__ == "__main__":
    main()
