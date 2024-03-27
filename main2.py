import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
from keras.models import load_model
ipl = pd.read_csv('ipl_data.csv')
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis =1)
X = df.drop(['total'], axis =1)
y = df['total']
# Create Tkinter window
from sklearn.preprocessing import LabelEncoder
# Create a LabelEncoder object for each categorical feature
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
batsman_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()
X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = batsman_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
model = load_model('model_saved.h5')
root = tk.Tk()
root.title("Cricket Score Predictor")
root.geometry("400x300")
# Create labels and dropdowns
venue_label = ttk.Label(root, text="Select Venue:")
venue_label.pack()
venue_var = tk.StringVar()
venue_dropdown = ttk.Combobox(root, textvariable=venue_var)
venue_dropdown.pack()
batting_label = ttk.Label(root, text="Select Batting Team:")
batting_label.pack()
batting_var = tk.StringVar()
batting_dropdown = ttk.Combobox(root, textvariable=batting_var)
batting_dropdown.pack()
bowling_label = ttk.Label(root, text="Select Bowling Team:")
bowling_label.pack()
bowling_var = tk.StringVar()
bowling_dropdown = ttk.Combobox(root, textvariable=bowling_var)
bowling_dropdown.pack()
batsman_label = ttk.Label(root, text="Select Batsman:")
batsman_label.pack()
batsman_var = tk.StringVar()
batsman_dropdown = ttk.Combobox(root, textvariable=batsman_var)
batsman_dropdown.pack()
bowler_label = ttk.Label(root, text="Select Bowler:")
bowler_label.pack()
bowler_var = tk.StringVar()
bowler_dropdown = ttk.Combobox(root, textvariable=bowler_var)
bowler_dropdown.pack()
venues =df['venue'].unique().tolist()  
batting_teams = df['bat_team'].unique().tolist()  
bowling_teams = df['bowl_team'].unique().tolist() 
batsmen =df['batsman'].unique().tolist() 
bowlers =df['bowler'].unique().tolist()
venue_dropdown['values'] = venues
batting_dropdown['values'] = batting_teams
bowling_dropdown['values'] = bowling_teams
batsman_dropdown['values'] = batsmen
bowler_dropdown['values'] = bowlers
# Function to predict score
def predict_score():
    # Get selected values
    selected_venue = venue_var.get()
    selected_batting_team = batting_var.get()
    selected_bowling_team = bowling_var.get()
    selected_batsman = batsman_var.get()
    selected_bowler = bowler_var.get()
    decoded_venue = venue_encoder.transform([selected_venue])
    decoded_batting_team = batting_team_encoder.transform([selected_batting_team])
    decoded_bowling_team = bowling_team_encoder.transform([selected_bowling_team])
    decoded_striker = batsman_encoder.transform([selected_batsman])
    decoded_bowler = bowler_encoder.transform([selected_bowler])
    # Prepare input data
    input_data = np.array([decoded_venue, decoded_batting_team, decoded_bowling_team, decoded_striker, decoded_bowler])
    input_data = input_data.reshape(1, 5)
    input_data = scaler.transform(input_data)
    # Predict score
    predicted_score = model.predict(input_data)
    predicted_score = int(predicted_score[0, 0])
    # Display predicted score
    result_label.config(text=f"Predicted Score: {predicted_score}")
# Button to trigger prediction
predict_button = ttk.Button(root, text="Predict Score", command=predict_score)
predict_button.pack()
result_label = ttk.Label(root, text="")
result_label.pack()
# Start Tkinter event loop
root.mainloop()
