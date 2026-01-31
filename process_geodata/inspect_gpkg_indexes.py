import sqlite3
import pandas as pd
import geopandas as gpd

# f = r"C:\Users\Elis\repos\us-datacentres\data\downloads\shapefiles\geoBoundariesCGAZ_ADM0.gpkg"
f = r"C:\Users\Elis\repos\us-datacentres\data\downloads\shapefiles\Countries_December_2024_Boundaries_UK_BUC_-4247675800557514417.gpkg"

conn = sqlite3.connect(f)
cursor = conn.cursor()

# List all indexes in the database
cursor.execute("""
    SELECT name, tbl_name, sql
    FROM sqlite_master
    WHERE type = 'index';
""")

indexes = cursor.fetchall()
for idx in indexes:
    print(idx)

# get table info
df = pd.read_sql_query(
    "PRAGMA table_info(globalADM0);",
    conn
)
conn.close()
print("schema:")
print(df)

# # read as gdf, see data
# gdf = gpd.read_file(f, layer="globalADM0")
# print("gdf.head():")
# print(gdf.head(10))


def inspect_gpkg(path):
    conn = sqlite3.connect(path)

    queries = {
        "tables": """
            SELECT type, name, tbl_name
            FROM sqlite_master
            WHERE type IN ('table', 'view');
        """,
        "layers": "SELECT * FROM gpkg_geometry_columns;",
        "indexes": """
            SELECT name, tbl_name, sql
            FROM sqlite_master
            WHERE type='index' AND sql IS NOT NULL;
        """
    }

    results = {
        name: pd.read_sql_query(sql, conn)
        for name, sql in queries.items()
    }

    conn.close()
    return results

info = inspect_gpkg(f)
print("Tables:")
print(info["tables"])
print("Layers:")
print(info["layers"])
print("Indexes:")
print(info["indexes"])

