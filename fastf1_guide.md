# FastF1 Guide: Functions, Abbreviations, and Usage

## 1. **Installation & Setup**
FastF1 is a Python library for extracting and analyzing Formula 1 race data.

### **Installation:**
```bash
pip install fastf1
```

### **Enable Caching (Recommended for Performance)**
```python
import fastf1
fastf1.Cache.enable_cache('cache_directory_path')
```

---

## 2. **Fetching Race Session Data**

### **`get_session()` - Fetching a specific session**
```python
session = fastf1.get_session(2024, 2, 'R')
```
- **Arguments:**
  - `year` (int): The season year (e.g., 2024)
  - `round` (int): The race number in the season (e.g., 2 for Saudi GP)
  - `session_type` (str): The type of session

### **Session Types:**
| Code  | Meaning |
|--------|----------------|
| `'FP1'` | Free Practice 1 |
| `'FP2'` | Free Practice 2 |
| `'FP3'` | Free Practice 3 |
| `'Q'` | Qualifying |
| `'SQ'` | Sprint Qualifying |
| `'S'` | Sprint Race |
| `'R'` | Race |

### **Loading the Session Data**
```python
session.load()
```
This function fetches all available race data, including drivers, teams, timings, and weather.

---

## 3. **Extracting Key Data**

### **Race Results**
```python
results = session.results
print(results[['Driver', 'TeamName', 'GridPosition', 'Position', 'Points']])
```

| Column | Meaning |
|--------|---------|
| `Driver` | Driver Name |
| `TeamName` | Team Name |
| `GridPosition` | Starting Position |
| `Position` | Final Position |
| `Points` | Points Earned |


### **Session Weather Data**
```python
weather = session.weather_data
print(weather[['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']])
```

| Column | Meaning |
|--------|---------|
| `AirTemp` | Air Temperature (°C) |
| `TrackTemp` | Track Temperature (°C) |
| `Humidity` | Humidity (%) |
| `Rainfall` | Rainfall (0 = no, 1 = yes) |


---

## 4. **Driver & Team Statistics**

### **Driver Standings**
```python
driver_standings = session.drivers
```

### **Team Standings**
```python
team_standings = session.results.groupby('TeamName').sum()['Points']
```

### **Filtering Data**
```python
filtered_results = results[results['GridPosition'] <= 10]  # Top 10 qualifiers
```

---

## 5. **Telemetry & Lap Data**
FastF1 allows you to analyze lap times, speeds, and telemetry data.

### **Fetching Lap Data for a Driver**
```python
laps = session.laps.pick_driver('VER')  # Data for Max Verstappen
print(laps[['LapNumber', 'LapTime', 'SpeedST']])
```

### **Telemetry Data (Speed, Throttle, Brake, etc.)**
```python
driver = session.drivers[0]  # First driver in the session
driver_laps = session.laps.pick_driver(driver)
fastest_lap = driver_laps.pick_fastest()
telemetry = fastest_lap.get_car_data()
print(telemetry[['Speed', 'Throttle', 'Brake']])
```

| Column | Meaning |
|--------|---------|
| `Speed` | Car Speed (km/h) |
| `Throttle` | Throttle Input (%) |
| `Brake` | Brake Input (0 or 1) |

---

## 6. **Common Abbreviations in FastF1**

| Abbreviation | Meaning |
|-------------|---------|
| `GridPosition` | Starting Position |
| `ClassifiedPosition` | Final Position (Includes Retirements) |
| `FastestLapTime` | Best Lap Time in the Race |
| `LapTime` | Time for each lap |
| `NumberOfPitStops` | Number of Pit Stops |
| `Points` | Championship Points Earned |
| `AirTemp` | Air Temperature |
| `TrackTemp` | Track Temperature |

---

## 7. **Additional Features**

### **Pit Stop Data**
```python
pit_stops = session.laps.pick_driver('HAM')[['LapNumber', 'PitOutTime', 'PitInTime']]
```

### **Qualifying Results**
```python
quali = fastf1.get_session(2024, 2, 'Q')
quali.load()
print(quali.results[['Driver', 'Q1', 'Q2', 'Q3']])
```

### **Sprint Race Results**
```python
sprint = fastf1.get_session(2024, 2, 'S')
sprint.load()
print(sprint.results[['Driver', 'Position', 'Points']])
```

---

## 8. **Conclusion**
This README provides the **fundamentals** of working with FastF1. You can:
- Fetch race sessions
- Extract results & standings
- Analyze driver & team performance
- Work with telemetry & lap data

For more, visit the [FastF1 Documentation](https://docs.fastf1.dev/).

