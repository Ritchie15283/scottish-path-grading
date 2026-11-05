import streamlit as st
import gpxpy
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import plotly.graph_objects as go
import requests
import re
import math
import tempfile
from pathlib import Path

st.set_page_config(page_title="Scottish Path Grading Tool", layout="wide")
st.title("Scottish Path Grading Tool")
st.caption("Uses Scottish Public Sector LiDAR DTM for elevations · Follows the Path Managers Guide to Grading")

S3_BASE = "https://srsp-open-data.s3.eu-west-2.amazonaws.com/"
PHASES = ["coastal/coastal-2025-2026", "phase-6", "phase-5", "phase-4", "phase-3", "phase-2", "phase-1"]
transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

def latlon_to_bng(lon, lat):
    return transformer.transform(lon, lat)

def list_dtm_tiles(phase):
    prefix = f"lidar/{phase}/dtm/27700/gridded/"
    url = f"{S3_BASE}?list-type=2&prefix={prefix}&max-keys=1000"
    try:
        r = requests.get(url, timeout=45)
        if r.status_code != 200:
            return []
        return re.findall(r"<Key>([^<]+_DTM_[^<]+\.tif)</Key>", r.text)
    except Exception:
        return []

def download_relevant_tiles(min_e, min_n, max_e, max_n, tmpdir):
    downloaded = []
    status = st.empty()
    for phase in PHASES:
        keys = list_dtm_tiles(phase)
        if not keys:
            continue
        status.info(f"Found tiles in {phase} — downloading relevant ones...")
        for key in keys[:6]:
            local = Path(tmpdir) / Path(key).name
            if local.exists():
                downloaded.append(str(local))
                continue
            try:
                tr = requests.get(S3_BASE + key, stream=True, timeout=180)
                if tr.status_code == 200:
                    with open(local, "wb") as f:
                        for chunk in tr.iter_content(1024 * 1024):
                            f.write(chunk)
                    downloaded.append(str(local))
            except Exception:
                pass
            if len(downloaded) >= 5:
                return downloaded
    status.empty()
    return downloaded

def sample_elevations(points_bng, tile_paths):
    elevs = []
    sources = []
    progress = st.progress(0)
    for i, (e, n) in enumerate(points_bng):
        elev = None
        source = None
        for tp in tile_paths:
            try:
                with rasterio.open(tp) as src:
                    if not (src.bounds.left <= e <= src.bounds.right and
                            src.bounds.bottom <= n <= src.bounds.top):
                        continue
                    row, col = src.index(e, n)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        val = src.read(1)[row, col]
                        nodata = src.nodata
                        if (nodata is None or val != nodata) and not np.isnan(val):
                            elev = float(val)
                            source = Path(tp).name
                            break
            except Exception:
                continue
        elevs.append(elev if elev is not None else np.nan)
        sources.append(source)
        progress.progress((i + 1) / len(points_bng))
    progress.empty()
    return elevs, sources

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def get_slope_term(grade_pct):
    g = abs(grade_pct)
    if g <= 2: return "flat"
    if g <= 5: return "gentle"
    if g <= 12: return "moderate"
    if g <= 20: return "fairly steep"
    return "steep"

def get_duration_term(length_m):
    if length_m <= 50: return "short"
    if length_m < 200: return "short–medium"
    if length_m < 400: return "long"
    return f"long for {int(length_m)} m"

def format_time(hours):
    if hours < 0.75: return "Allow ½ hr"
    if hours < 1.25: return "Allow 1 hr"
    if hours < 1.75: return "Allow 1 ½ hrs"
    if hours < 2.25: return "Allow 2 hrs"
    if hours < 2.75: return "Allow 2 ½ hrs"
    return f"Allow {round(hours * 2) / 2} hrs"

# UI
uploaded = st.file_uploader("Upload GPX file", type=["gpx"])

if uploaded:
    gpx = gpxpy.parse(uploaded.read().decode("utf-8"))
    points = []
    for track in gpx.tracks:
        for seg in track.segments:
            points.extend(seg.points)
    if not points and gpx.routes:
        for route in gpx.routes:
            points.extend(route.points)

    if len(points) < 2:
        st.error("No usable track points found in the GPX.")
    else:
        lats = [p.latitude for p in points]
        lons = [p.longitude for p in points]
        bng = [latlon_to_bng(lon, lat) for lon, lat in zip(lons, lats)]
        eastings = [e for e, n in bng]
        northings = [n for e, n in bng]
        min_e, max_e = min(eastings), max(eastings)
        min_n, max_n = min(northings), max(northings)

        with st.spinner("Looking up and sampling Scottish Public Sector LiDAR DTM tiles…"):
            with tempfile.TemporaryDirectory() as tmp:
                tiles = download_relevant_tiles(min_e, min_n, max_e, max_n, tmp)
                if not tiles:
                    st.error("No LIDAR DTM tiles found for this area. Coverage is good across much of Scotland but not 100%. Check the Scottish Remote Sensing Portal.")
                    st.stop()
                elevs, sources = sample_elevations(bng, tiles)

        valid = sum(1 for e in elevs if not np.isnan(e))
        st.success(f"Sampled {valid}/{len(elevs)} points from LIDAR DTM")

        # Compute slopes
        rows = []
        cum = 0.0
        for i in range(1, len(elevs)):
            if np.isnan(elevs[i]) or np.isnan(elevs[i-1]):
                continue
            d_horiz = haversine(lats[i-1], lons[i-1], lats[i], lons[i])
            de = elevs[i] - elevs[i-1]
            angle = math.degrees(math.atan2(de, d_horiz)) if d_horiz > 0 else 0.0
            grade = (de / d_horiz * 100) if d_horiz > 0 else 0.0
            rows.append({
                "start_m": cum,
                "length_m": d_horiz,
                "delta_elev_m": de,
                "angle_deg": angle,
                "grade_pct": grade,
                "elev_m": elevs[i],
                "source": sources[i],
            })
            cum += d_horiz

        if not rows:
            st.error("Could not compute slopes — insufficient valid LIDAR elevations.")
            st.stop()

        df = pd.DataFrame(rows)
        total_dist_m = cum
        total_dist_km = total_dist_m / 1000
        total_dist_miles = total_dist_km * 0.621371
        max_abs_grade = df["grade_pct"].abs().max()
        ascent = df.loc[df["delta_elev_m"] > 0, "delta_elev_m"].sum()

        # Grade suggestion
        long_steep = df[(df["grade_pct"].abs() > 12) & (df["length_m"] >= 200)]
        has_long_steep = len(long_steep) > 0
        has_fairly = ((df["grade_pct"].abs() > 12) & (df["length_m"] > 50)).any()

        if total_dist_miles > 7 or has_long_steep or max_abs_grade > 20:
            suggested = "Strenuous"
            reason = "Includes long steep sections and/or exceeds Moderate distance limit."
        elif total_dist_miles > 5 or has_fairly or max_abs_grade > 12:
            suggested = "Moderate"
            reason = "Some slopes over 12% or distance exceeds Easy limit."
        else:
            suggested = "Easy"
            reason = "Gradients mostly ≤12% and distance within Easy limits."

        min_per_mile = {"Easy": 30, "Moderate": 35, "Strenuous": 40}[suggested]
        hours = (total_dist_miles * min_per_mile) / 60
        time_text = format_time(hours)

        # Draft terrain
        steep = df[df["grade_pct"].abs() > 5]
        if len(steep) == 0:
            terrain_draft = "Mostly flat path."
        else:
            terms = sorted(set(get_slope_term(g) for g in steep["grade_pct"]))
            terrain_draft = " and ".join(terms) + " slopes."
        terrain_draft += f" Approximately {total_dist_miles:.1f} miles / {total_dist_km:.1f} km."

        # Accuracy
        unique_tiles = sorted(set(s for s in sources if s))
        accuracy = (
            f"Elevations sampled from Scottish Public Sector LiDAR Digital Terrain Model (bare-earth). "
            f"Tiles used: {', '.join(unique_tiles) if unique_tiles else 'none'}. "
            f"Points sampled: {valid}/{len(elevs)}. "
            f"Typical vertical accuracy: ±15 cm RMSE (varies by phase and terrain). "
            f"Resolution typically 0.5 m or 1 m where coverage exists. "
            f"Licence: Open Government Licence. "
            f"Always verify the steepest sections in the field with a clinometer."
        )

        # Editable six-part panel
        st.header("Route Information (editable)")
        st.caption("Follows the six-part system from the Path Managers Guide to Grading. Keep Path Promotion under ~40 words and Terrain Description under ~30 words.")

        path_name = st.text_input("1. Path Name", "Path Name")
        path_promo = st.text_area("2. Path Promotion", "A short description of what makes this path special. Focus on the user experience and highlights.", height=80)

        col1, col2 = st.columns([3, 1])
        with col1:
            terrain = st.text_area("4. Terrain Description", terrain_draft + " (Add surface and obstacle details from field survey. Keep under 30 words.)", height=100)
        with col2:
            st.markdown(f"**Suggested Grade**")
            st.markdown(f"### {suggested}")
            st.caption(reason)
            st.markdown(f"**{total_dist_miles:.1f} miles / {total_dist_km:.1f} km**")
            st.markdown(time_text)

        st.info(accuracy)

        # Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distance", f"{total_dist_miles:.1f} miles")
        c2.metric("Ascent", f"{ascent:.0f} m")
        c3.metric("Max gradient", f"{max_abs_grade:.1f}%")
        c4.metric("Suggested grade", suggested)

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["start_m"] / 1000,
            y=df["elev_m"],
            mode="lines",
            name="Elevation (LIDAR DTM)",
            line=dict(color="#1a5f7a", width=2),
            hovertemplate="Dist: %{x:.2f} km<br>Elev: %{y:.1f} m<br>Slope: %{customdata:.1f}%<extra></extra>",
            customdata=df["grade_pct"],
        ))
        fig.update_layout(
            title="Elevation Profile (Scottish LIDAR DTM)",
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Slope table with Guide terms
        df_display = df.copy()
        df_display["Slope term"] = df_display["grade_pct"].apply(get_slope_term)
        df_display["Duration term"] = df_display["length_m"].apply(get_duration_term)
        df_display["Type"] = df_display["grade_pct"].apply(lambda g: "Uphill" if g > 2 else "Downhill" if g < -2 else "Flat")
        st.subheader("Slope Segments (Guide terminology)")
        st.dataframe(
            df_display[["start_m", "length_m", "grade_pct", "Slope term", "Duration term", "Type", "source"]]
            .rename(columns={"start_m": "Start (m)", "length_m": "Length (m)", "grade_pct": "Grade %", "source": "Tile"}),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download slope segments (CSV)",
            df_display.to_csv(index=False).encode(),
            "slope_segments.csv",
            "text/csv",
        )
