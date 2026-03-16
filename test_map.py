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

    return df, agency_df, tracts


df, agency_df, tracts = load_data()


# ==========================================================
# CLEAN DATA
# ==========================================================

df["tractid"] = df["tractid"].astype(str).str.zfill(11)
tracts["GEOID"] = tracts["GEOID"].astype(str)

gdf = tracts.merge(df, left_on="GEOID", right_on="tractid", how="inner")

gdf = gdf.to_crs(epsg=4326)

gdf["LI/LA"] = gdf["LI/LA"].astype(str)


# ==========================================================
# AGENCY POINT DATA
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


def get_lila_color(val):

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
# MAP TYPE SELECTOR
# ==========================================================

st.subheader("SNAP / LI-LA Map")

map_mode = st.selectbox(
    "Select map visualization",
    ["SNAP Bivariate Classification", "LI/LA Classification"]
)


# ==========================================================
# SNAP MAP LOGIC
# ==========================================================

if map_mode == "SNAP Bivariate Classification":

    acs_year = st.selectbox("Select ACS Data Year", ["2022","2023"])

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

else:

    selected = st.multiselect(
        "Select LI/LA classification",
        options=["1","0","Not In Data"],
        default=["1","0","Not In Data"]
    )

    filtered_gdf = gdf[gdf["LI/LA"].isin(selected)].copy()
    filtered_gdf["color"] = filtered_gdf["LI/LA"].apply(get_lila_color)


# ==========================================================
# MAP 1
# ==========================================================

m = folium.Map(location=[36.05,-79.9], zoom_start=7, tiles="cartodbpositron")

def style_function(feature):
    return {
        "fillColor": feature["properties"]["color"],
        "color": "black",
        "weight": 0.3,
        "fillOpacity": 0.7
    }

tooltip_fields = ["County","tractid","Agency Count","Average Increaase in Visit"]

folium.GeoJson(
    filtered_gdf,
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=["County","Tract","Agency Count","Visit Change"]
    )
).add_to(m)

# Agency markers
for _,row in agency_gdf.iterrows():

    folium.CircleMarker(
        location=[row["lat"],row["long"]],
        radius=1.5,
        color="black",
        weight=0.5,
        fill=True,
        fill_color="#1f77b4",
        fill_opacity=0.9,
        tooltip=row["Agency Short Name"]
    ).add_to(m)

st_folium(m, height=750, use_container_width=True)


# ==========================================================
# VISIT CHANGE MAP
# ==========================================================

st.subheader("Visit Change Map")

gdf["change_color"] = gdf["Average Increaase in Visit"].map(change_colors)

m2 = folium.Map(location=[36.05,-79.9], zoom_start=7, tiles="cartodbpositron")

def style_change(feature):
    return {
        "fillColor": feature["properties"]["change_color"],
        "color": "black",
        "weight": 0.3,
        "fillOpacity": 0.7
    }

folium.GeoJson(
    gdf,
    style_function=style_change,
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "County",
            "tractid",
            "Agency Count",
            "Average Increaase in Visit",
            "Need Level 2023"
        ],
        aliases=[
            "County:",
            "Tract:",
            "Agency Count:",
            "Visit Change:",
            "Need Level:"
        ],
        sticky=True
    )
).add_to(m2)

for _,row in agency_gdf.iterrows():

    folium.CircleMarker(
        location=[row["lat"],row["long"]],
        radius=1.5,
        color="black",
        weight=0.5,
        fill=True,
        fill_color="#1f77b4",
        fill_opacity=0.9,
        tooltip=row["Agency Short Name"]
    ).add_to(m2)
# ----------------------------------------------------------
# VISIT CHANGE LEGEND
# ----------------------------------------------------------

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
# NEED LEVEL MAP
# ==========================================================

st.subheader("Food Access Need Map")

need_colors = {
    "Has Agency": "#7790b3",
    "Neighboring Agency": "#9ac2bf",
    "High Need": "#d77c7b",
    "Moderate Need": "#e8a663"
}

gdf["need_color"] = gdf["Need Level 2023"].map(need_colors)

m3 = folium.Map(
    location=[36.05, -79.9],
    zoom_start=7,
    tiles="cartodbpositron"
)

# ----------------------------------------------------------
# STYLE FUNCTION
# ----------------------------------------------------------

def style_need(feature):

    return {
        "fillColor": feature["properties"]["need_color"],
        "color": "black",
        "weight": 0.3,
        "fillOpacity": 0.7
    }


# ----------------------------------------------------------
# HOVER TOOLTIP FOR TRACTS
# ----------------------------------------------------------

tooltip_fields = [
    "County",
    "tractid",
    "Agency Count",
    "SNAP Participant Count 2023",
    "Above SNAP Median 2023",
    "Need Level 2023"
]

tooltip_alias = [
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
        fields=tooltip_fields,
        aliases=tooltip_alias,
        sticky=True,
        labels=True
    )
).add_to(m3)


# ----------------------------------------------------------
# AGENCY POINT OVERLAY
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# LEGEND
# ----------------------------------------------------------

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
# ==========================================================
# SNAP VS LI/LA ANALYSIS
# ==========================================================

st.subheader("SNAP vs LI/LA Analysis")

if map_mode == "SNAP Bivariate Classification":

    pivot_table = pd.crosstab(
        gdf["LI/LA"],
        gdf[formulation_col]
    )

    pivot_table["Total"] = pivot_table.sum(axis=1)

    total_row = pivot_table.sum(axis=0)
    total_row.name = "Total"

    pivot_table = pd.concat([pivot_table,total_row.to_frame().T])

    st.dataframe(pivot_table)

    # Prepare chart data
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
        legend=alt.Legend(
            title="LI/LA Status",
            orient="bottom"
        )
    ),
        tooltip=["SNAP Category","LI/LA","Count"]
    ).properties(
        width=700,
        height=350
    )

    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.altair_chart(chart)


# ==========================================================
# INCREASE IN VISIT ANALYSIS
# ==========================================================

st.subheader("Increase in Visit Analysis")

increase_df = gdf[gdf["Average Increaase in Visit"] == "Increase"]

st.write("Number of tracts with increased visits:", len(increase_df))

# ----------------------------------------------------------
# CREATE PIVOT
# ----------------------------------------------------------

pivot_inc = pd.crosstab(
    increase_df["LI/LA"],
    increase_df[formulation_col]
)

# add totals
pivot_inc["Total"] = pivot_inc.sum(axis=1)

total_row = pivot_inc.sum(axis=0)
total_row.name = "Total"

pivot_inc = pd.concat([pivot_inc, total_row.to_frame().T])

# ----------------------------------------------------------
# TRANSPOSE FOR DISPLAY
# ----------------------------------------------------------

pivot_display = pivot_inc.T

st.write("Pivot Table (Increase in Visits Only)")
st.dataframe(pivot_display)

# ----------------------------------------------------------
# PREP DATA FOR ALTAIR
# ----------------------------------------------------------

plot_df = pivot_display.drop("Total").drop(columns="Total")

chart_df = plot_df.reset_index()

# rename first column safely
chart_df = chart_df.rename(columns={chart_df.columns[0]: "SNAP Category"})

chart_df = chart_df.melt(
    id_vars="SNAP Category",
    var_name="LI/LA",
    value_name="Count"
)

# ----------------------------------------------------------
# ALTAIR CHART
# ----------------------------------------------------------

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
        legend=alt.Legend(
            title="LI/LA Status",
            orient="bottom"
        )
    ),
    tooltip=["SNAP Category","LI/LA","Count"]
).properties(
    width=700,
    height=350
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.altair_chart(chart2)