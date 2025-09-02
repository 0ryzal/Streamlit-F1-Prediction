import fastf1
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load assets
stack_model = joblib.load("model/f1_race_predictor_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
filtered_drivers_info = pd.read_csv("model/DATA/filtered_drivers_info.csv")

# Get driver list
driver_abbrs = filtered_drivers_info["Abbreviation"].tolist()

# Get 2024 schedule from FastF1
schedule = fastf1.get_event_schedule(2024)
schedule = schedule.drop(0)
event_names = schedule['EventName'].tolist()
event_rounds = schedule['RoundNumber'].tolist()
race_name_to_round = dict(zip(event_names, event_rounds))

# Streamlit UI setup
st.set_page_config(page_title="F1 Predictor", layout="wide")
st.title("🏎️ F1 Race Predictor")
st.markdown("Select the race and enter driver grid positions to predict the final standings.")

# Dropdown for race selection
selected_race_name = st.selectbox("Select Race", event_names)
round_number = race_name_to_round[selected_race_name]

# Grid positions input (driver selection instead of numbers)
st.subheader("Grid Positions (1 = Pole Position)")

grid_positions = {}
chosen_drivers = set()

cols = st.columns(4)
for pos in range(1, 21):  # 20 grid positions
    with cols[(pos - 1) % 4]:
        # Filter supaya driver yang sudah dipilih tidak muncul lagi
        available = [d for d in driver_abbrs if d not in chosen_drivers]
        driver = st.selectbox(
            f"Grid Position {pos}",
            options=["-"] + available,
            key=f"grid_pos_{pos}"
        )
        if driver != "-":
            grid_positions[pos] = driver
            chosen_drivers.add(driver)


# Prediction logic
if st.button("Predict Race Results"):
    # Balik mapping: driver -> posisi grid
    driver_to_grid = {driver: pos for pos, driver in grid_positions.items() if driver != "-"}

    # Pastikan semua driver sudah ditempatkan
    if len(driver_to_grid) < 20:
        st.error("⚠️ Semua driver harus dipilih sebelum prediksi!")
    else:
        GridPosition = [driver_to_grid[driver] for driver in driver_abbrs]

        pred_gp_data = pd.DataFrame({
            "Round": [round_number] * 20,
            "Abbreviation": driver_abbrs,
            "GridPosition": GridPosition,
            "Points": filtered_drivers_info["Points"],
            "AvgQualiPosition": filtered_drivers_info["AvgQualiPosition"],
            "AvgRacePosition": filtered_drivers_info["AvgRacePosition"],
            "QualifyingScore": (filtered_drivers_info["AvgQualiPosition"] + GridPosition) / 2
        })

        # ... lanjut ke scaler + prediksi seperti sebelumnya


    label_enc_driver = LabelEncoder()
    pred_gp_data["Abbreviation"] = label_enc_driver.fit_transform(pred_gp_data["Abbreviation"])

    pred_gp_data = pred_gp_data[feature_columns]
    X_scaled = scaler.transform(pred_gp_data)
    predicted_positions = stack_model.predict(X_scaled)
    pred_gp_data["PredictedPosition"] = predicted_positions

    results = pred_gp_data.sort_values("PredictedPosition").reset_index(drop=True)
    results.index += 1
    results.rename_axis("PredictedRank", inplace=True)
    results = results.reset_index()
    results["Driver_Abbreviation"] = label_enc_driver.inverse_transform(results["Abbreviation"])

    st.success(f"📊 Predicted Results for {selected_race_name} (Round {round_number})")
    st.dataframe(results[["PredictedRank", "Driver_Abbreviation"]], use_container_width=True)
