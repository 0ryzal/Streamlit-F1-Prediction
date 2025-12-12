import fastf1
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ===== LOAD MODELS & DATA =====
stack_model = joblib.load("model/f1_race_predictor_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
filtered_drivers_info = pd.read_csv("model/DATA/filtered_drivers_info.csv")

# Driver list
driver_abbrs = filtered_drivers_info["Abbreviation"].tolist()
driver_full_names = dict(zip(filtered_drivers_info["Abbreviation"], filtered_drivers_info["DriverName"]))

# Team colors (F1 official 2024)
TEAM_COLORS = {
    "VER": "#3671C6", "PER": "#3671C6",  # Red Bull
    "HAM": "#27F4D2", "RUS": "#27F4D2",  # Mercedes
    "LEC": "#E8002D", "SAI": "#E8002D",  # Ferrari
    "NOR": "#FF8000", "PIA": "#FF8000",  # McLaren
    "ALO": "#229971", "STR": "#229971",  # Aston Martin
    "GAS": "#5E8FAA", "OCO": "#5E8FAA",  # Alpine
    "BOT": "#52E252", "ZHO": "#52E252",  # Kick Sauber
    "TSU": "#6692FF", "RIC": "#6692FF",  # VCARB
    "ALB": "#64C4FF", "SAR": "#64C4FF",  # Williams
    "MAG": "#B6BABD", "HUL": "#B6BABD",  # Haas
}

# Get 2024 schedule
schedule = fastf1.get_event_schedule(2024)
schedule = schedule.drop(0)
event_names = schedule['EventName'].tolist()
event_rounds = schedule['RoundNumber'].tolist()
race_name_to_round = dict(zip(event_names, event_rounds))

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="F1 Race Predictor 2024",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CUSTOM CSS (F1 OFFICIAL STYLE) =====
st.markdown("""
    <style>
    /* Import F1 Font */
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Titillium Web', sans-serif !important;
    }
    
    /* Main Background - F1 Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #15151E 0%, #1a1a2e 50%, #15151E 100%);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #E10600 0%, #FF1E00 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(225, 6, 0, 0.3);
    }
    
    .main-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: 0.5rem;
    }
    
    /* Results Table Styling */
    .results-table {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .position-row {
        display: flex;
        align-items: center;
        padding: 1rem;
        margin: 0.5rem 0;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    
    .position-row:hover {
        background: rgba(255,255,255,0.1);
        transform: translateX(5px);
    }
    
    .position-number {
        font-size: 1.8rem;
        font-weight: 900;
        color: white;
        width: 50px;
        text-align: center;
    }
    
    .driver-info {
        flex: 1;
        margin-left: 1.5rem;
    }
    
    .driver-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .driver-abbr {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.6);
        font-weight: 400;
    }
    
    .grid-pos {
        font-size: 1rem;
        color: rgba(255,255,255,0.7);
        background: rgba(0,0,0,0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 5px;
        font-weight: 600;
    }
    
    .points-badge {
        background: linear-gradient(135deg, #E10600 0%, #FF1E00 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-left: 1rem;
    }
    
    /* Podium Special Styling */
    .position-1 { border-left-color: #FFD700 !important; }
    .position-2 { border-left-color: #C0C0C0 !important; }
    .position-3 { border-left-color: #CD7F32 !important; }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #E10600 0%, #FF1E00 100%);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 0.8rem 3rem;
        border: none;
        border-radius: 5px;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(225, 6, 0, 0.6);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 5px;
        color: white;
    }
    
    /* Section Headers */
    .section-header {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #E10600;
        padding-left: 1rem;
    }
    
    /* Race Info Card */
    .race-info-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .race-info-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    
    .race-info-value {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Grid Position Input */
    .grid-input-section {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Confidence Indicator */
    .confidence-bar {
        width: 100%;
        height: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #E10600 0%, #FFD700 100%);
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🏎️ F1 Race Predictor</h1>
        <p class="subtitle">Predict race finishing positions using advanced machine learning</p>
    </div>
""", unsafe_allow_html=True)

# ===== RACE SELECTION =====
st.markdown('<div class="section-header">📍 Race Selection</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    selected_race_name = st.selectbox("Select Grand Prix", event_names, label_visibility="collapsed")
    round_number = race_name_to_round[selected_race_name]

with col2:
    st.markdown(f"""
        <div class="race-info-card">
            <div class="race-info-label">Round</div>
            <div class="race-info-value">{round_number}</div>
        </div>
    """, unsafe_allow_html=True)

# ===== GRID POSITIONS INPUT =====
st.markdown('<div class="section-header">🏁 Grid Positions</div>', unsafe_allow_html=True)
st.markdown('<div class="grid-input-section">', unsafe_allow_html=True)

grid_positions = {}
chosen_drivers = set()

# Create 5 columns for 4 drivers each (20 total)
for row in range(5):
    cols = st.columns(4)
    for col_idx in range(4):
        pos = row * 4 + col_idx + 1
        if pos <= 20:
            with cols[col_idx]:
                available = [d for d in driver_abbrs if d not in chosen_drivers]
                driver = st.selectbox(
                    f"P{pos}",
                    options=[""] + available,
                    key=f"grid_pos_{pos}",
                    format_func=lambda x: f"{x} - {driver_full_names.get(x, '')}" if x else "Select Driver"
                )
                if driver:
                    grid_positions[pos] = driver
                    chosen_drivers.add(driver)

st.markdown('</div>', unsafe_allow_html=True)

# ===== PREDICTION BUTTON =====
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    predict_button = st.button("🏆 PREDICT RACE", use_container_width=True)

# ===== PREDICTION LOGIC =====
if predict_button:
    driver_to_grid = {driver: pos for pos, driver in grid_positions.items() if driver}
    
    if len(driver_to_grid) < 20:
        st.error("⚠️ Please select all 20 drivers before prediction!")
    else:
        with st.spinner("🔄 Analyzing race data and predicting results..."):
            # Prepare data with enhanced features
            GridPosition = [driver_to_grid.get(driver, 20) for driver in driver_abbrs]
            
            # Enhanced feature engineering
            driver_info = filtered_drivers_info.set_index("Abbreviation")
            
            pred_data = []
            for i, driver in enumerate(driver_abbrs):
                grid_pos = GridPosition[i]
                driver_stats = driver_info.loc[driver]
                
                # Calculate advanced features
                quali_score = (driver_stats["AvgQualiPosition"] + grid_pos) / 2
                grid_improvement = driver_stats["AvgRacePosition"] - grid_pos
                consistency_score = abs(driver_stats["AvgQualiPosition"] - driver_stats["AvgRacePosition"])
                
                pred_data.append({
                    "Round": round_number,
                    "Abbreviation": driver,
                    "GridPosition": grid_pos,
                    "Points": driver_stats["Points"],
                    "AvgQualiPosition": driver_stats["AvgQualiPosition"],
                    "AvgRacePosition": driver_stats["AvgRacePosition"],
                    "QualifyingScore": quali_score,
                })
            
            pred_gp_data = pd.DataFrame(pred_data)
            
            # Encode drivers
            label_enc_driver = LabelEncoder()
            pred_gp_data["Abbreviation_Encoded"] = label_enc_driver.fit_transform(pred_gp_data["Abbreviation"])
            
            # Prepare features for prediction
            feature_data = pred_gp_data.copy()
            feature_data["Abbreviation"] = feature_data["Abbreviation_Encoded"]
            feature_data = feature_data[feature_columns]
            
            # Scale and predict
            X_scaled = scaler.transform(feature_data)
            predicted_positions = stack_model.predict(X_scaled)
            
            # Create results dataframe
            pred_gp_data["PredictedPosition"] = predicted_positions
            pred_gp_data["Confidence"] = 100 - (abs(predicted_positions - pred_gp_data["GridPosition"]) * 3)
            pred_gp_data["Confidence"] = pred_gp_data["Confidence"].clip(50, 100)
            
            results = pred_gp_data.sort_values("PredictedPosition").reset_index(drop=True)
            results.index += 1
            
            # Calculate points based on F1 point system
            points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            results["PredictedPoints"] = results.index.map(lambda x: points_system.get(x, 0))
            
            # ===== DISPLAY RESULTS =====
            st.markdown('<div class="section-header">🏆 Predicted Race Results</div>', unsafe_allow_html=True)
            
            # Top 3 Podium Highlight
            st.markdown("### 🥇 Podium")
            podium_cols = st.columns(3)
            
            for idx, (_, row) in enumerate(results.head(3).iterrows()):
                with podium_cols[idx]:
                    position_emoji = ["🥇", "🥈", "🥉"][idx]
                    driver_abbr = row["Abbreviation"]
                    driver_name = driver_full_names[driver_abbr]
                    team_color = TEAM_COLORS.get(driver_abbr, "#FFFFFF")
                    
                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {team_color}40 0%, {team_color}20 100%);
                                    border: 2px solid {team_color};
                                    border-radius: 10px;
                                    padding: 1.5rem;
                                    text-align: center;">
                            <div style="font-size: 3rem;">{position_emoji}</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: white; margin-top: 0.5rem;">
                                {driver_name}
                            </div>
                            <div style="font-size: 1rem; color: rgba(255,255,255,0.7); margin-top: 0.3rem;">
                                {driver_abbr}
                            </div>
                            <div style="font-size: 1.2rem; color: {team_color}; font-weight: 700; margin-top: 0.8rem;">
                                {int(row['PredictedPoints'])} PTS
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Full Results Table
            st.markdown('<div class="results-table">', unsafe_allow_html=True)
            
            for idx, (_, row) in enumerate(results.iterrows(), 1):
                driver_abbr = row["Abbreviation"]
                driver_name = driver_full_names[driver_abbr]
                team_color = TEAM_COLORS.get(driver_abbr, "#FFFFFF")
                grid_pos = int(row["GridPosition"])
                pts = int(row["PredictedPoints"])
                confidence = int(row["Confidence"])
                
                position_class = f"position-{idx}" if idx <= 3 else ""
                
                st.markdown(f"""
                    <div class="position-row {position_class}" style="border-left-color: {team_color};">
                        <div class="position-number">{idx}</div>
                        <div class="driver-info">
                            <div class="driver-name">{driver_name}</div>
                            <div class="driver-abbr">{driver_abbr}</div>
                        </div>
                        <div class="grid-pos">Grid: P{grid_pos}</div>
                        {f'<div class="points-badge">{pts} PTS</div>' if pts > 0 else ''}
                        <div style="margin-left: 1rem; width: 100px;">
                            <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">Confidence</div>
                            <div style="color: white; font-weight: 700;">{confidence}%</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence}%;"></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # ===== VISUALIZATION =====
            st.markdown('<div class="section-header">📊 Position Changes</div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            
            for _, row in results.iterrows():
                driver_abbr = row["Abbreviation"]
                team_color = TEAM_COLORS.get(driver_abbr, "#FFFFFF")
                
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[row["GridPosition"], row.name],
                    mode='lines+markers+text',
                    name=driver_abbr,
                    line=dict(color=team_color, width=3),
                    marker=dict(size=12, color=team_color),
                    text=[driver_abbr, driver_abbr],
                    textposition=['middle left', 'middle right'],
                    textfont=dict(size=10, color=team_color, family='Titillium Web'),
                    hovertemplate=f'{driver_full_names[driver_abbr]}<br>Grid: P{int(row["GridPosition"])}<br>Predicted: P{row.name}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Grid Position → Predicted Finish",
                xaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1],
                    ticktext=['Grid', 'Predicted'],
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    title='Position',
                    autorange='reversed',
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    zeroline=False
                ),
                showlegend=False,
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Titillium Web'),
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown('<div class="section-header">📈 Race Statistics</div>', unsafe_allow_html=True)
            
            stat_cols = st.columns(4)
            
            biggest_gainer = results.loc[(results["GridPosition"] - results.index).idxmax()]
            biggest_loser = results.loc[(results["GridPosition"] - results.index).idxmin()]
            avg_confidence = results["Confidence"].mean()
            
            with stat_cols[0]:
                positions_gained = int(biggest_gainer["GridPosition"] - biggest_gainer.name)
                st.markdown(f"""
                    <div class="race-info-card">
                        <div class="race-info-label">Biggest Gainer</div>
                        <div class="race-info-value">{biggest_gainer["Abbreviation"]}</div>
                        <div style="color: #00FF00; font-size: 1rem;">↑ {positions_gained} positions</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat_cols[1]:
                positions_lost = int(biggest_loser.name - biggest_loser["GridPosition"])
                st.markdown(f"""
                    <div class="race-info-card">
                        <div class="race-info-label">Biggest Loser</div>
                        <div class="race-info-value">{biggest_loser["Abbreviation"]}</div>
                        <div style="color: #FF0000; font-size: 1rem;">↓ {positions_lost} positions</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat_cols[2]:
                st.markdown(f"""
                    <div class="race-info-card">
                        <div class="race-info-label">Avg Confidence</div>
                        <div class="race-info-value">{avg_confidence:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat_cols[3]:
                st.markdown(f"""
                    <div class="race-info-card">
                        <div class="race-info-label">Total Drivers</div>
                        <div class="race-info-value">20</div>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.5); padding: 2rem;">
        <p>Powered by FastF1 & Machine Learning | Data from 2024 Season</p>
    </div>
""", unsafe_allow_html=True)
