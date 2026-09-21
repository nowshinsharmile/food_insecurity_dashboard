# ==========================================================
# STREAMLIT FOOD ACCESS DASHBOARD
# ==========================================================

import streamlit as st
st.set_page_config(layout="wide", page_title="Bivariate Classification Visualization")

import geopandas as gpd
import pandas as pd
import folium
import altair as alt
from streamlit_folium import st_folium
import numpy as np
import requests
import base64
from datetime import datetime


# ==========================================================
# FILE PATHS
# ==========================================================

excel_file = "SNAP_Bivariate_Classification_Dataset_v2.xlsx"
shapefile = "cb_2023_37_tract_500k/cb_2023_37_tract_500k.shp"


## ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    df = pd.read_excel(excel_file, sheet_name="Sheet4")
    agency_df = pd.read_excel(excel_file, sheet_name="Agency_Data_v2")
    tracts = gpd.read_file(shapefile)

    # Clean column names from Excel
    df.columns = df.columns.str.strip()
    agency_df.columns = agency_df.columns.str.strip()
    tracts.columns = tracts.columns.str.strip()

    # Optional: simplify geometry for faster rendering
    tracts["geometry"] = tracts["geometry"].simplify(0.0005)

    return df, agency_df, tracts


df, agency_df, tracts = load_data()


# ==========================================================
# CLEAN, EXCLUDE, AND RECALCULATE DATA
# ==========================================================

required_columns = [
    "tractid",
    "County",
    "Agency Count",
    "SNAP Participant Count 2022",
    "SNAP Participant Count 2023",
    "Average Increase in Visit",
    "LI/LA",
    "Excluded from Service Region"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing required columns in Sheet4: {', '.join(missing_columns)}")
    st.stop()

df["tractid"] = df["tractid"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)
tracts["GEOID"] = tracts["GEOID"].astype(str)

excluded_flag = (
    df["Excluded from Service Region"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["1", "1.0", "true", "yes", "y"])
)
excluded_tract_count = int(excluded_flag.sum())
df = df.loc[~excluded_flag].copy()

gdf = tracts.merge(df, left_on="GEOID", right_on="tractid", how="inner").to_crs(epsg=4326)

if gdf.empty:
    st.error("No tracts remain after applying the service-region exclusion field.")
    st.stop()

gdf["Agency Count"] = pd.to_numeric(gdf["Agency Count"], errors="coerce").fillna(0)
gdf["Agency Presence"] = np.where(gdf["Agency Count"] > 0, "Agency Presence", "No Agency Presence")
gdf["Agency Presency"] = (gdf["Agency Count"] > 0).astype(int)


def calculate_neighbor_agency_status(service_gdf):
    """Return True when a tract without an agency touches an included tract that has one."""
    agency_present = service_gdf["Agency Count"].gt(0)
    spatial_index = service_gdf.sindex
    output = pd.Series(False, index=service_gdf.index, dtype=bool)

    for idx, geometry in service_gdf.geometry.items():
        if agency_present.loc[idx] or geometry is None or geometry.is_empty:
            continue
        candidate_positions = spatial_index.query(geometry, predicate="touches")
        candidate_indices = service_gdf.index.take(candidate_positions)
        candidate_indices = candidate_indices[candidate_indices != idx]
        output.loc[idx] = bool(agency_present.reindex(candidate_indices, fill_value=False).any())

    return output


gdf["Neighboring Agency Coverage"] = calculate_neighbor_agency_status(gdf)


def recalculate_year_fields(service_gdf, year):
    """Recalculate all year-dependent classifications from included tracts only."""
    snap_col = f"SNAP Participant Count {year}"
    above_col = f"Above SNAP Median {year}"
    formulation_col_year = f"Formulation {year}"
    need_col = f"Need Level {year}"

    service_gdf[snap_col] = pd.to_numeric(service_gdf[snap_col], errors="coerce")
    snap_median = service_gdf[snap_col].median(skipna=True)
    has_snap = service_gdf[snap_col].notna()
    above_median = has_snap & service_gdf[snap_col].gt(snap_median)

    service_gdf[above_col] = np.select(
        [~has_snap, above_median],
        ["Not Available", "Above SNAP Median"],
        default="Below SNAP Median"
    )
    service_gdf[formulation_col_year] = np.select(
        [
            ~has_snap,
            above_median & service_gdf["Agency Count"].eq(0),
            above_median & service_gdf["Agency Count"].gt(0),
            ~above_median & service_gdf["Agency Count"].eq(0)
        ],
        [
            "Not Available",
            "Above SNAP Median,No Agency Presence",
            "Above SNAP Median,Agency Presence",
            "Below SNAP Median,No Agency Presence"
        ],
        default="Below SNAP Median,Agency Presence"
    )
    service_gdf[need_col] = np.select(
        [
            service_gdf["Agency Count"].gt(0),
            service_gdf["Neighboring Agency Coverage"],
            ~has_snap,
            above_median
        ],
        ["Has Agency", "Neighboring Agency", "Not Available", "High Need"],
        default="Moderate Need"
    )

    return snap_median


snap_medians = {year: recalculate_year_fields(gdf, year) for year in ["2022", "2023"]}
gdf["Above SNAP and No Agency Coverage"] = (
    gdf["Above SNAP Median 2022"].eq("Above SNAP Median") &
    gdf["Agency Count"].eq(0)
).astype(int)
gdf["Quadrant"] = gdf["Formulation 2022"]

# Fix mixed type issues
gdf["LI/LA"] = gdf["LI/LA"].astype(str).str.strip().str.lower()
# ==========================================================
# FIX LI/LA LABELS (ONLY CHANGE)
# ==========================================================
gdf["LI/LA"] = gdf["LI/LA"].replace({
    "1": "LI/LA",
    "1.0": "LI/LA",
    "0": "Not LI/LA",
    "0.0": "Not LI/LA",
    "nan": "Not In Data",
    "none": "Not In Data",
    "not in data": "Not In Data",
    "not in database": "Not In Data"
})
gdf["Average Increase in Visit"] = gdf["Average Increase in Visit"].fillna("No Agency").astype(str).str.strip()

# Need level columns may or may not exist for both years
for col in ["Need Level 2022", "Need Level 2023"]:
    if col in gdf.columns:
        gdf[col] = gdf[col].astype(str)


# ==========================================================
# CREATE AGENCY POINT DATA FROM Agency_Data_v2
# ==========================================================

def find_column(frame, candidates):
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None

lat_col = find_column(agency_df, ["lat", "latitude", "agency_lat", "agency latitude"])
lon_col = find_column(agency_df, ["long", "lon", "lng", "longitude", "agency_long", "agency longitude"])

if lat_col is None or lon_col is None:
    st.error("Agency_Data_v2 must contain latitude and longitude columns. Accepted names include lat/latitude and long/lon/lng/longitude.")
    st.stop()

agency_df[lat_col] = pd.to_numeric(agency_df[lat_col], errors="coerce")
agency_df[lon_col] = pd.to_numeric(agency_df[lon_col], errors="coerce")
agency_df = agency_df.dropna(subset=[lat_col, lon_col]).copy()

if "Mapped_Tract_ID" in agency_df.columns:
    agency_df["Mapped_Tract_ID"] = agency_df["Mapped_Tract_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)

agency_gdf = gpd.GeoDataFrame(agency_df, geometry=gpd.points_from_xy(agency_df[lon_col], agency_df[lat_col]), crs="EPSG:4326")

try:
    service_area = gdf.geometry.union_all()
except AttributeError:
    service_area = gdf.geometry.unary_union

agency_gdf = agency_gdf[agency_gdf.geometry.intersects(service_area)].copy()

def agency_tooltip_html(row):
    parts = []
    if "Agency_Name" in row.index and pd.notna(row["Agency_Name"]):
        parts.append(f"<b>Agency:</b> {row['Agency_Name']}")
    elif "Agency_ID" in row.index and pd.notna(row["Agency_ID"]):
        parts.append(f"<b>Agency:</b> {row['Agency_ID']}")
    if "Program_Type" in row.index and pd.notna(row["Program_Type"]):
        parts.append(f"<b>Program:</b> {row['Program_Type']}")
    if "County" in row.index and pd.notna(row["County"]):
        parts.append(f"<b>County:</b> {row['County']}")
    if "Mapped_Tract_ID" in row.index and pd.notna(row["Mapped_Tract_ID"]):
        parts.append(f"<b>Mapped tract:</b> {row['Mapped_Tract_ID']}")
    return "<br>".join(parts) if parts else "Agency"

# ==========================================================
# COLOR DEFINITIONS
# ==========================================================

snap_colors = {
    # Two-color family with four shades
    # Above SNAP median = red shades
    "Above SNAP Median,No Agency Presence": "#8b0000",   # deep red
    "Above SNAP Median,Agency Presence": "#f4a6a6",      # light red

    # Below SNAP median = green shades
    "Below SNAP Median,No Agency Presence": "#d9f2d9",   # very light green
    "Below SNAP Median,Agency Presence": "#00a651"       # pure green
}

change_colors = {
    "Increase": "#93c883",
    "Decrease": "#e28980",
    "No Change": "#d5d487",
    "No Agency": "#bdbdbd"   # grey instead of pink
}

need_colors = {
    "Has Agency": "#7790b3",
    "Neighboring Agency": "#9ac2bf",
    "High Need": "#d77c7b",
    "Moderate Need": "#e8a663"
}


# ==========================================================
# MAP BOUNDS HELPER
# ==========================================================

def fit_map_to_gdf(m, map_gdf):
    """
    Fit Folium map to the displayed service area so the map does not load too zoomed out/small.
    """
    if map_gdf is None or map_gdf.empty:
        return m

    minx, miny, maxx, maxy = map_gdf.total_bounds

    # Folium expects [[south, west], [north, east]]
    m.fit_bounds([
        [miny, minx],
        [maxy, maxx]
    ])

    return m
def get_lila_color(val: str) -> str:
    val = str(val).strip()
    if val.lower() in ["not in data", "not in database"]:
        return "#e0e0e0"
    if val == "LI/LA":
        return "#8b0000"   # LI/LA = deep red
    return "#00a651"       # Not LI/LA = pure green

def save_feedback_to_github(name: str, comment: str) -> None:
    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]
    file_path = "feedback.txt"

    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }

    # Read current file if it exists
    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code == 200:
        payload = response.json()
        existing_content = base64.b64decode(payload["content"]).decode("utf-8")
        sha = payload["sha"]
    elif response.status_code == 404:
        existing_content = ""
        sha = None
    else:
        raise RuntimeError(f"GitHub read failed: {response.status_code} - {response.text}")

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    safe_name = name.strip() if name and name.strip() else "Anonymous"
    safe_comment = comment.strip()

    new_entry = (
        f"Time: {timestamp}\n"
        f"Name: {safe_name}\n"
        f"Comment: {safe_comment}\n"
        f"{'-'*50}\n"
    )

    updated_content = existing_content + new_entry
    encoded_content = base64.b64encode(updated_content.encode("utf-8")).decode("utf-8")

    data = {
        "message": "Append dashboard feedback",
        "content": encoded_content,
        "branch": "main"
    }

    if sha is not None:
        data["sha"] = sha

    write_response = requests.put(url, headers=headers, json=data, timeout=30)

    if write_response.status_code not in (200, 201):
        raise RuntimeError(f"GitHub write failed: {write_response.status_code} - {write_response.text}")
# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("Bivariate Classification Visualization")
st.caption(
    f"Service-region calculations use {len(gdf):,} included tracts. "
    f"{excluded_tract_count:,} tract(s) marked 'Excluded from Service Region = 1' were removed. "
    f"Recalculated SNAP medians — 2022: {snap_medians['2022']:,.0f}; 2023: {snap_medians['2023']:,.0f}."
)


# ==========================================================
# MAP 1 : SNAP / LI-LA / SNAP POPULATION
# ==========================================================

st.subheader("SNAP / LI-LA Map")

map_mode = st.selectbox(
    "Select map visualization",
    [
        "SNAP Bivariate Classification",
        "LI/LA Classification",
        "SNAP Population"
    ]
)

# ----------------------------------------------------------
# CHOOSE DATA MODE
# ----------------------------------------------------------

formulation_col = None
filtered_gdf = gdf.copy()

if map_mode == "SNAP Bivariate Classification":
    acs_year = st.selectbox("Select ACS Data Year", ["2022", "2023"])

    if acs_year == "2022":
        formulation_col = "Formulation 2022"
    else:
        formulation_col = "Formulation 2023"

    selected = st.multiselect(
        "Select bivariate classification",
        options=list(snap_colors.keys()),
        default=list(snap_colors.keys())
    )

    filtered_gdf = gdf[gdf[formulation_col].isin(selected)].copy()
    filtered_gdf["color"] = filtered_gdf[formulation_col].map(snap_colors)

elif map_mode == "SNAP Population":
    snap_year = st.selectbox("Select SNAP Year", ["2022", "2023"])
    snap_col = f"SNAP Participant Count {snap_year}"
    filtered_gdf = gdf.copy()

else:
    selected = st.multiselect(
        "Select LI/LA classification",
        options=["LI/LA", "Not LI/LA", "Not In Data"],
        default=["LI/LA", "Not LI/LA", "Not In Data"]  
    )

    filtered_gdf = gdf[gdf["LI/LA"].isin(selected)].copy()
    filtered_gdf["color"] = filtered_gdf["LI/LA"].apply(get_lila_color)


# ----------------------------------------------------------
# BUILD MAP 1
# ----------------------------------------------------------

m = folium.Map(tiles="OpenStreetMap")
m = fit_map_to_gdf(m, filtered_gdf)

# ==========================================================
# SNAP POPULATION (HEAT MAP)
# ==========================================================

if map_mode == "SNAP Population":

    values = filtered_gdf[snap_col].fillna(0)

    # ----------------------------------------------------------
    # CREATE QUANTILE BINS (8 colors)
    # ----------------------------------------------------------
    bins = np.quantile(values, np.linspace(0, 1, 9))
    bins = np.unique(bins)

    if len(bins) < 3:
        bins = np.linspace(values.min(), values.max(), 5)

    # ----------------------------------------------------------
    # COLOR PALETTE
    # ----------------------------------------------------------
    colors = [
        "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
        "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"
    ]

    # ----------------------------------------------------------
    # FUNCTION TO MAP VALUE → COLOR
    # ----------------------------------------------------------
    def get_color(val):
        for i in range(len(bins) - 1):
            if bins[i] <= val <= bins[i + 1]:
                return colors[min(i, len(colors)-1)]
        return colors[-1]

    # assign color column
    filtered_gdf["snap_color"] = filtered_gdf[snap_col].apply(get_color)

    # ----------------------------------------------------------
    # DRAW MAP (GeoJson instead of Choropleth)
    # ----------------------------------------------------------
    folium.GeoJson(
        filtered_gdf,
        style_function=lambda feature: {
            "fillColor": feature["properties"]["snap_color"],
            "color": "black",
            "weight": 0.2,
            "fillOpacity": 0.8
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "County",
                "tractid",
                "SNAP Participant Count 2022",
                "SNAP Participant Count 2023"
            ],
            aliases=[
                "County:",
                "Tract:",
                "SNAP 2022:",
                "SNAP 2023:"
            ],
            sticky=True
        )
    ).add_to(m)

    # ----------------------------------------------------------
    # CUSTOM LEGEND
    # ----------------------------------------------------------
    bin_labels = []
    for i in range(len(bins) - 1):
        low = int(bins[i])
        high = int(bins[i + 1])
        bin_labels.append(f"{low:,} – {high:,}")

    legend_items = ""
    for color, label in zip(colors[:len(bin_labels)], bin_labels):
        legend_items += f"""
        <div style="margin-bottom:4px;">
            <i style="background:{color};
                      width:15px;height:15px;
                      display:inline-block;margin-right:6px;"></i>
            {label}
        </div>
        """

    legend_html = f"""
    <div style="
    position: fixed;
    bottom: 30px; left: 40px;
    width: 260px;
    background:white;
    border:2px solid grey;
    z-index:9999;
    font-size:13px;
    padding:10px;
    ">

    <b>SNAP Population ({snap_year})</b><br>
    <span style="font-size:11px;">
    Quantile-based distribution
    </span><br><br>

    {legend_items}

    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

# ==========================================================
# SNAP BIVARIATE + LI/LA MAP
# ==========================================================

else:

    def style_function(feature):
        return {
            "fillColor": feature["properties"]["color"],
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.7
        }

    folium.GeoJson(
        filtered_gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["County", "tractid", "Agency Count", "Average Increase in Visit"],
            aliases=["County:", "Tract:", "Agency Count:", "Visit Change:"],
            sticky=True
        )
    ).add_to(m)

    # ------------------ Legends ------------------
    if map_mode == "SNAP Bivariate Classification":
        legend_html = """
        <div style="
        position: fixed;
        bottom: 30px; left: 40px;
        width: 260px;
        background:white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding:10px;
        ">

        <b>SNAP Bivariate Classification</b><br>

        <i style="background:#8b0000;width:15px;height:15px;display:inline-block"></i>
        Above SNAP Median, No Agency<br>

        <i style="background:#d9f2d9;width:15px;height:15px;display:inline-block"></i>
        Below SNAP Median, No Agency<br>

        <i style="background:#00a651;width:15px;height:15px;display:inline-block"></i>
        Below SNAP Median, Agency<br>

        <i style="background:#f4a6a6;width:15px;height:15px;display:inline-block"></i>
        Above SNAP Median, Agency

        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    elif map_mode == "LI/LA Classification":
        legend_html = """
        <div style="
        position: fixed;
        bottom: 30px; left: 40px;
        width: 220px;
        background:white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding:10px;
        ">

        <b>LI/LA Classification</b><br>

        <i style="background:#e5513f;width:15px;height:15px;display:inline-block"></i>
        LI/LA<br>

        <i style="background:#defd93;width:15px;height:15px;display:inline-block"></i>
        Not LI/LA<br>

        <i style="background:#e0e0e0;width:15px;height:15px;display:inline-block"></i>
        Not in Data

        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))


# ==========================================================
# ADD AGENCY POINTS (ONCE, BEFORE RENDER)
# ==========================================================

for _, row in agency_gdf.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=1.8,
        color="black",
        weight=0.5,
        fill=True,
        fill_color="#1f77b4",
        fill_opacity=0.9,
        tooltip=folium.Tooltip(agency_tooltip_html(row), sticky=True)
    ).add_to(m)


# ==========================================================
# RENDER MAP 1 (ONLY ONCE)
# ==========================================================

st_folium(m, height=750, use_container_width=True, returned_objects=[], key="main_map")

# ==========================================================
# MAP 2 : VISIT CHANGE MAP
# ==========================================================

st.subheader("Visit Change Map")

gdf["change_color"] = gdf["Average Increase in Visit"].map(change_colors).fillna("#cccccc")
m2 = folium.Map(tiles="OpenStreetMap")
m2 = fit_map_to_gdf(m2, gdf)

def style_change(feature):
    return {
        "fillColor": feature["properties"]["change_color"],
        "color": "black",
        "weight": 0.3,
        "fillOpacity": 0.7}


visit_tooltip_fields = ["County", "tractid", "Agency Count", "Average Increase in Visit"]
visit_tooltip_aliases = ["County:", "Tract:", "Agency Count:", "Visit Change:"]

if "Need Level 2023" in gdf.columns:
    visit_tooltip_fields.append("Need Level 2023")
    visit_tooltip_aliases.append("Need Level:")

folium.GeoJson(
    gdf,
    style_function=style_change,
    tooltip=folium.GeoJsonTooltip(
        fields=visit_tooltip_fields,
        aliases=visit_tooltip_aliases,
        sticky=True)).add_to(m2)

for _, row in agency_gdf.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=1.8,
        color="black",
        weight=0.5,
        fill=True,
        fill_color="#1f77b4",
        fill_opacity=0.9,
        tooltip=folium.Tooltip(agency_tooltip_html(row), sticky=True)
    ).add_to(m2)

legend2 = """
<div style="
position: fixed; 
bottom: 40px; left: 40px; 
width: 220px;
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:14px;
padding: 10px;
">

<b>Visit Change</b><br>

<i style="background:#93c883;width:15px;height:15px;display:inline-block"></i>
Increase<br>

<i style="background:#e28980;width:15px;height:15px;display:inline-block"></i>
Decrease<br>

<i style="background:#d5d487;width:15px;height:15px;display:inline-block"></i>
No Change<br>


<i style="background:#bdbdbd;width:15px;height:15px;display:inline-block"></i>
No Agency

</div>
"""

m2.get_root().html.add_child(folium.Element(legend2))
st_folium(m2, height=750, use_container_width=True, returned_objects=[], key="visit_change_map")


# ==========================================================
# MAP 3 : NEED LEVEL MAP
# ==========================================================

st.subheader("Food Access Need Map")

need_level_col = "Need Level 2023" if "Need Level 2023" in gdf.columns else None

if need_level_col:
    gdf["need_color"] = gdf[need_level_col].map(need_colors).fillna("#cccccc")

    m3 = folium.Map(tiles="OpenStreetMap")
    m3 = fit_map_to_gdf(m3, gdf)

    def style_need(feature):
        return {
            "fillColor": feature["properties"]["need_color"],
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.7
        }

    need_tooltip_fields = [
        "County",
        "tractid",
        "Agency Count",
        "SNAP Participant Count 2023",
        "Above SNAP Median 2023",
        need_level_col
    ]

    need_tooltip_aliases = [
        "County:",
        "Tract:",
        "Agency Count:",
        "SNAP Participants:",
        "SNAP Median:",
        "Need Level:"
    ]

    folium.GeoJson(
        gdf,
        style_function=style_need,
        tooltip=folium.GeoJsonTooltip(
            fields=need_tooltip_fields,
            aliases=need_tooltip_aliases,
            sticky=True,
            labels=True
        )
    ).add_to(m3)

    for _, row in agency_gdf.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=2,
            color="black",
            weight=0.5,
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.9,
            tooltip=folium.Tooltip(agency_tooltip_html(row), sticky=True)
        ).add_to(m3)

    legend3 = """
    <div style="
    position: fixed; 
    bottom: 40px; left: 40px; 
    width: 230px;
    background:white;
    border:2px solid grey;
    z-index:9999;
    font-size:14px;
    padding:10px;
    ">

    <b>Food Access Need</b><br>

    <i style="background:#7790b3;width:15px;height:15px;display:inline-block"></i>
    Has Agency<br>

    <i style="background:#9ac2bf;width:15px;height:15px;display:inline-block"></i>
    Neighboring Agency<br>

    <i style="background:#d77c7b;width:15px;height:15px;display:inline-block"></i>
    High Need<br>

    <i style="background:#e8a663;width:15px;height:15px;display:inline-block"></i>
    Moderate Need

    </div>
    """

    m3.get_root().html.add_child(folium.Element(legend3))
    st_folium(m3, height=750, use_container_width=True, returned_objects=[], key="need_level_map")
else:
    st.info("Need level columns were not found in the uploaded data.")


st.subheader("SNAP vs LI/LA Analysis")

# ----------------------------------------------------------
# ALWAYS SET DEFAULT
# ----------------------------------------------------------
if formulation_col is None:
    formulation_col = "Formulation 2023"

if formulation_col in gdf.columns:

    pivot_table = pd.crosstab(
        gdf["LI/LA"],
        gdf[formulation_col]
    )

    pivot_table["Total"] = pivot_table.sum(axis=1)

    total_row = pivot_table.sum(axis=0)
    total_row.name = "Total"

    pivot_table = pd.concat([pivot_table, total_row.to_frame().T])

    st.dataframe(pivot_table)



# ==========================================================
# INCREASE IN VISIT ANALYSIS
# ==========================================================

if formulation_col is not None:
    st.subheader("Increase in Visit Analysis")

    increase_df = gdf[gdf["Average Increase in Visit"] == "Increase"]

    st.write("Number of tracts with increased visits:", len(increase_df))

    pivot_inc = pd.crosstab(
        increase_df["LI/LA"],
        increase_df[formulation_col]
    )

    pivot_inc["Total"] = pivot_inc.sum(axis=1)

    total_row = pivot_inc.sum(axis=0)
    total_row.name = "Total"

    pivot_inc = pd.concat([pivot_inc, total_row.to_frame().T])

    pivot_display = pivot_inc.T

    st.write("Pivot Table (Increase in Visits Only)")
    st.dataframe(pivot_display)

    plot_df = pivot_display.drop("Total").drop(columns="Total")

    chart_df = plot_df.reset_index()
    chart_df = chart_df.rename(columns={chart_df.columns[0]: "SNAP Category"})

    chart_df = chart_df.melt(
        id_vars="SNAP Category",
        var_name="LI/LA",
        value_name="Count"
    )

st.divider()
st.subheader("Agency Analysis")

agency_id_count = agency_gdf["Agency_ID"].nunique() if "Agency_ID" in agency_gdf.columns else len(agency_gdf)
agency_counties = agency_gdf["County"].nunique() if "County" in agency_gdf.columns else 0
program_type_count = agency_gdf["Program_Type"].nunique() if "Program_Type" in agency_gdf.columns else 0
metric1, metric2, metric3 = st.columns(3)
metric1.metric("Agencies plotted", f"{agency_id_count:,}")
metric2.metric("Counties represented", f"{agency_counties:,}")
metric3.metric("Program types", f"{program_type_count:,}")

if "County" in agency_gdf.columns:
    agency_by_county = agency_gdf.groupby("County").size().reset_index(name="Agency Count from Agency_Data_v2").sort_values("Agency Count from Agency_Data_v2", ascending=False)
    st.markdown("#### Agencies by County")
    st.dataframe(agency_by_county, use_container_width=True, hide_index=True)

if "Program_Type" in agency_gdf.columns:
    agency_by_program = agency_gdf.groupby("Program_Type").size().reset_index(name="Agency Count").sort_values("Agency Count", ascending=False)
    st.markdown("#### Agencies by Program Type")
    st.dataframe(agency_by_program, use_container_width=True, hide_index=True)

if "Mapped_Tract_ID" in agency_gdf.columns:
    point_counts = agency_gdf.groupby("Mapped_Tract_ID").size().rename("Agency Count from Agency_Data_v2").reset_index()
    tract_check = gdf[["tractid", "County", "Agency Count"]].drop_duplicates("tractid").merge(point_counts, left_on="tractid", right_on="Mapped_Tract_ID", how="left")
    tract_check["Agency Count from Agency_Data_v2"] = tract_check["Agency Count from Agency_Data_v2"].fillna(0).astype(int)
    tract_check["Agency Count"] = pd.to_numeric(tract_check["Agency Count"], errors="coerce").fillna(0).astype(int)
    tract_check["Difference"] = tract_check["Agency Count from Agency_Data_v2"] - tract_check["Agency Count"]
    mismatches = tract_check[tract_check["Difference"] != 0].copy()
    st.markdown("#### Agency Count Validation by Census Tract")
    st.caption("Compares point-level agencies from Agency_Data_v2 against Sheet4 Agency Count using Mapped_Tract_ID.")
    check1, check2 = st.columns(2)
    check1.metric("Tracts matching exactly", f"{int((tract_check['Difference'] == 0).sum()):,}")
    check2.metric("Tracts with count mismatch", f"{len(mismatches):,}")
    if mismatches.empty:
        st.success("Agency_Data_v2 point counts match Sheet4 Agency Count for every included tract.")
    else:
        st.dataframe(mismatches[["tractid", "County", "Agency Count", "Agency Count from Agency_Data_v2", "Difference"]].sort_values(["County", "tractid"]), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Leave Feedback")

with st.form("feedback_form"):
    user_name = st.text_input("Name (optional)")
    user_comment = st.text_area("Your feedback")
    submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        if not user_comment.strip():
            st.warning("Please enter a comment before submitting.")
        else:
            try:
                save_feedback_to_github(user_name, user_comment)
                st.success("Thanks — your feedback was submitted.")
            except Exception as e:
                st.error(f"Feedback could not be saved. {e}")
