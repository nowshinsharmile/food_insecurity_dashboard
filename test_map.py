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


# ==========================================================
# FILE PATHS
# ==========================================================

excel_file = "SNAP_Bivariate _Classification_Dataset.xlsx"
shapefile = "cb_2023_37_tract_500k/cb_2023_37_tract_500k.shp"


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    df = pd.read_excel(excel_file, sheet_name="Sheet3")
    agency_df = pd.read_excel(excel_file, sheet_name="Agency_Data")
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
# CLEAN DATA
# ==========================================================

df["tractid"] = df["tractid"].astype(str).str.zfill(11)
tracts["GEOID"] = tracts["GEOID"].astype(str)

gdf = tracts.merge(df, left_on="GEOID", right_on="tractid", how="inner")
gdf = gdf.to_crs(epsg=4326)

# Fix mixed type issues
gdf["LI/LA"] = gdf["LI/LA"].astype(str)
gdf["Average Increaase in Visit"] = gdf["Average Increaase in Visit"].astype(str)

# Need level columns may or may not exist for both years
for col in ["Need Level 2022", "Need Level 2023"]:
    if col in gdf.columns:
        gdf[col] = gdf[col].astype(str)


# ==========================================================
# CREATE AGENCY POINT DATA
# ==========================================================

agency_gdf = gpd.GeoDataFrame(
    agency_df,
    geometry=gpd.points_from_xy(agency_df["long"], agency_df["lat"]),
    crs="EPSG:4326"
)


# ==========================================================
# COLOR DEFINITIONS
# ==========================================================

snap_colors = {
    "Above SNAP Median,No Agency Presence": "#ea524a",
    "Below SNAP Median,No Agency Presence": "#6ecffa",
    "Below SNAP Median,Agency Presence": "#7dba53",
    "Above SNAP Median,Agency Presence": "#f9dd5f"
}

change_colors = {
    "Increase": "#93c883",
    "Decrease": "#e28980",
    "No Change": "#d5d487",
    "No Agency": "#e5acd0"
}

need_colors = {
    "Has Agency": "#7790b3",
    "Neighboring Agency": "#9ac2bf",
    "High Need": "#d77c7b",
    "Moderate Need": "#e8a663"
}


def get_lila_color(val: str) -> str:
    val = str(val).strip()
    if val.lower() in ["not in data", "not in database"]:
        return "#e0e0e0"
    if val == "1":
        return "#e5513f"
    return "#defd93"


# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("Bivariate Classification Visualization")


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
        options=["1", "0", "Not In Data"],
        default=["1", "0", "Not In Data"]
    )

    filtered_gdf = gdf[gdf["LI/LA"].isin(selected)].copy()
    filtered_gdf["color"] = filtered_gdf["LI/LA"].apply(get_lila_color)


# ----------------------------------------------------------
# BUILD MAP 1
# ----------------------------------------------------------

m = folium.Map(location=[36.05, -79.9], zoom_start=7, tiles="cartodbpositron")

if map_mode == "SNAP Population":

    import numpy as np

    # ----------------------------------------------------------
    # CREATE QUANTILE BINS (8 colors)
    # ----------------------------------------------------------
    values = filtered_gdf[snap_col].fillna(0)

    bins = np.quantile(values, np.linspace(0, 1, 9))
    bins = np.unique(bins)

    # fallback if bins collapse (edge case)
    if len(bins) < 3:
        bins = np.linspace(values.min(), values.max(), 5)

    # ----------------------------------------------------------
    # CHOROPLETH
    # ----------------------------------------------------------
    folium.Choropleth(
        geo_data=filtered_gdf,
        data=filtered_gdf,
        columns=["tractid", snap_col],
        key_on="feature.properties.tractid",
        fill_color="YlOrRd",
        bins=bins,
        fill_opacity=0.8,
        line_opacity=0.2,
        legend_name=None,  # disable broken default legend
        nan_fill_color="lightgray"
    ).add_to(m)

    # ----------------------------------------------------------
    # CUSTOM LEGEND
    # ----------------------------------------------------------
    bin_labels = []
    for i in range(len(bins) - 1):
        low = int(bins[i])
        high = int(bins[i + 1])
        bin_labels.append(f"{low:,} – {high:,}")

    colors = [
        "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
        "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"
    ]

    # match number of colors to bins
    colors = colors[:len(bin_labels)]

    legend_items = ""
    for color, label in zip(colors, bin_labels):
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
    Relative distribution (each color ≈ equal number of tracts)
    </span><br><br>

    {legend_items}

    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    # ----------------------------------------------------------
    # HOVER LAYER
    # ----------------------------------------------------------
    folium.GeoJson(
        filtered_gdf,
        style_function=lambda x: {
            "fillOpacity": 0,
            "color": "black",
            "weight": 0.2
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


# ==========================================================
# MAP 2 : VISIT CHANGE MAP
# ==========================================================

st.subheader("Visit Change Map")

gdf["change_color"] = gdf["Average Increaase in Visit"].map(change_colors).fillna("#cccccc")

m2 = folium.Map(location=[36.05, -79.9], zoom_start=7, tiles="cartodbpositron")


def style_change(feature):
    return {
        "fillColor": feature["properties"]["change_color"],
        "color": "black",
        "weight": 0.3,
        "fillOpacity": 0.7
    }


visit_tooltip_fields = ["County", "tractid", "Agency Count", "Average Increaase in Visit"]
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
        sticky=True
    )
).add_to(m2)

for _, row in agency_gdf.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=1.8,
        color="black",
        weight=0.5,
        fill=True,
        fill_color="#1f77b4",
        fill_opacity=0.9,
        tooltip=f"Agency: {row['Agency Short Name']}"
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

<i style="background:#e5acd0;width:15px;height:15px;display:inline-block"></i>
No Agency

</div>
"""

m2.get_root().html.add_child(folium.Element(legend2))
st_folium(m2, height=750, use_container_width=True)


# ==========================================================
# MAP 3 : NEED LEVEL MAP
# ==========================================================

st.subheader("Food Access Need Map")

need_level_col = "Need Level 2023" if "Need Level 2023" in gdf.columns else None

if need_level_col:
    gdf["need_color"] = gdf[need_level_col].map(need_colors).fillna("#cccccc")

    m3 = folium.Map(
        location=[36.05, -79.9],
        zoom_start=7,
        tiles="cartodbpositron"
    )

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
            location=[row["lat"], row["long"]],
            radius=2,
            color="black",
            weight=0.5,
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.9,
            tooltip=folium.Tooltip(
                f"""
                <b>Agency:</b> {row['Agency Short Name']}<br>
                <b>Lat:</b> {row['lat']}<br>
                <b>Lon:</b> {row['long']}
                """,
                sticky=True
            )
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
    st_folium(m3, height=750, use_container_width=True)
else:
    st.info("Need level columns were not found in the uploaded data.")


# ==========================================================
# SNAP VS LI/LA ANALYSIS
# ==========================================================

st.subheader("SNAP vs LI/LA Analysis")

if map_mode == "SNAP Bivariate Classification" and formulation_col is not None:
    pivot_table = pd.crosstab(
        gdf["LI/LA"],
        gdf[formulation_col]
    )

    pivot_table["Total"] = pivot_table.sum(axis=1)

    total_row = pivot_table.sum(axis=0)
    total_row.name = "Total"

    pivot_table = pd.concat([pivot_table, total_row.to_frame().T])

    st.dataframe(pivot_table)

    plot_df = pivot_table.drop("Total").drop(columns="Total")

    chart_df = plot_df.T.reset_index()
    chart_df.columns = ["SNAP Category"] + list(chart_df.columns[1:])

    chart_df = chart_df.melt(
        id_vars="SNAP Category",
        var_name="LI/LA",
        value_name="Count"
    )

    st.subheader("Distribution of SNAP Categories by LI/LA Status")

    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X(
            "SNAP Category:N",
            sort=None,
            title="SNAP Classification",
            axis=alt.Axis(labelAngle=-25)
        ),
        y=alt.Y("Count:Q", title="Number of Census Tracts"),
        color=alt.Color(
            "LI/LA:N",
            legend=alt.Legend(title="LI/LA Status", orient="bottom")
        ),
        tooltip=["SNAP Category", "LI/LA", "Count"]
    ).properties(
        width=700,
        height=350
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.altair_chart(chart)


# ==========================================================
# INCREASE IN VISIT ANALYSIS
# ==========================================================

if formulation_col is not None:
    st.subheader("Increase in Visit Analysis")

    increase_df = gdf[gdf["Average Increaase in Visit"] == "Increase"]

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

    chart2 = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X(
            "SNAP Category:N",
            sort=None,
            title="SNAP Classification",
            axis=alt.Axis(labelAngle=-25)
        ),
        y=alt.Y("Count:Q", title="Number of Census Tracts"),
        color=alt.Color(
            "LI/LA:N",
            legend=alt.Legend(title="LI/LA Status", orient="bottom")
        ),
        tooltip=["SNAP Category", "LI/LA", "Count"]
    ).properties(
        width=700,
        height=350
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.altair_chart(chart2)