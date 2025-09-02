# 🏎️ Formula 1 Race Predictor

A machine learning project that predicts the **finishing order of all 20 drivers** in a Formula 1 race based on historical data, driver stats, and current grid positions. Built using FastF1, Scikit-Learn, XGBoost, and Streamlit.

---

## 📦 Features
- Predict full race finishing positions
- Supports all 2024 races (auto-loaded from FastF1)
- Inputs: Grid position per driver
- Feature engineered with:
  - Average grid & race positions
  - Driver & team experience
  - Total Points from 2024
  - Qualifying score
- Deployable via Streamlit Cloud

---

## 🚀 How It Works

### 1. **Model**
- Trained with: `StackingRegressor`
- Base estimators: Ridge, Lasso, XGBoost, GradientBoosting
- Final estimator: Ridge or Lasso
- Feature scaling: `StandardScaler`
- Saved as: `stack_model.pkl`, `scaler.pkl`, `feature_columns.pkl`

### 2. **Data**
- Source: [FastF1](https://docs.fastf1.dev/)
- Final dataset: `f1_final_data.csv`, merged with driver/team stats
- Manually created files:
  - `filtered_drivers_info.csv`

### 3. **Streamlit App**
- UI includes:
  - Dropdown to select race
  - Input grid positions for all 20 drivers
  - Display predicted order

---

## 🛠️ Setup

### 🔗 Requirements
```bash
pip install -r requirements.txt
```

### 📂 Folder Structure
```
Formula1-Predictor/
├── fastf1_guide.md               # Guides to all commands in fastf1
├── app.py                        # Streamlit app
├── collect_f1_data.ipynb         # Collects the data from fastf1
├── build_features.ipynb          # Making the final dataset for the model
├── train_model.ipynb             # Main training file of the model
├── model/
│   ├── f1_race_predictor_model.pkl        # Trained model
│   ├── scaler.pkl                         # Scaler
│   └── feature_columns.pkl                # Column order
├── DATA/
│   ├── f1_results_2024_2025.csv           # Combined race data
│   ├── f1_drivers_points_exp.xlsx         # Driver's Experience
│   ├── f1_final_data.csv                  # Final dataset for model training
│   └── filtered_drivers_info.csv          # Final dataset for driver's information
└── requirements.txt
```

---

## 🌐 Live App
You can try the live app here:  
**[https://formula1-predcition-model.streamlit.app/](https://formula1-predcition-model.streamlit.app/)**

---

## 🌍 Deployment
1. Push this project to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repo, select `app.py`
4. Click **Deploy**

---
## 📊 Usage

```bash
streamlit run app.py
```
- Select a race (2024 calendar)
- Enter grid positions for each driver
- Click **Predict** to see results

---
