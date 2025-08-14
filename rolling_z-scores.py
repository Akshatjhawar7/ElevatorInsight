import pandas as pd
import matplotlib.pyplot as plt
import pathlib

csv_path = pathlib.Path("data/predictive-maintenance-dataset.csv")
df = pd.read_csv(csv_path)

#Setting rolling size window
window = 50

def rolling_zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series-mean)/std

#Calculating rolling z-scores
df['vibration_z'] = rolling_zscore(df['vibration'], window)
df['revolutions_z'] = rolling_zscore(df['revolutions'], window)

#Plotting
plt.figure(figsize=(14,6))
plt.plot(df['vibration_z'], label='Vibration Z-Score', alpha=0.7)
plt.plot(df['revolutions_z'], label='Revolutions Z-Score', alpha=0.7)
plt.axhline(3, color='red', linestyle='--', linewidth=1)
plt.axhline(-3, color='red', linestyle='--', linewidth=1)
plt.title('Rolling Z-Scores (window = {})'.format(window))
plt.xlabel('Time Index')
plt.ylabel('Z-Score')
plt.legend()
plt.show()