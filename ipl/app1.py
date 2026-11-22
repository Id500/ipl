from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --------------------------------
# LOAD DATASET
# --------------------------------

data = pd.read_csv("Match_Info.csv")

# Keep needed columns
data = data[['team1',
             'team2',
             'toss_winner',
             'toss_decision',
             'venue',
             'winner']]

# Remove null values
data.dropna(inplace=True)

# --------------------------------
# LABEL ENCODERS
# --------------------------------

team_encoder = LabelEncoder()
toss_encoder = LabelEncoder()
venue_encoder = LabelEncoder()
winner_encoder = LabelEncoder()

# Fit team encoder with all team columns
all_teams = pd.concat([
    data['team1'],
    data['team2'],
    data['toss_winner'],
    data['winner']
])

team_encoder.fit(all_teams)

# Fit other encoders
toss_encoder.fit(data['toss_decision'])

venue_encoder.fit(data['venue'])

winner_encoder.fit(data['winner'])

# Transform columns
data['team1'] = team_encoder.transform(data['team1'])

data['team2'] = team_encoder.transform(data['team2'])

data['toss_winner'] = team_encoder.transform(data['toss_winner'])

data['toss_decision'] = toss_encoder.transform(data['toss_decision'])

data['venue'] = venue_encoder.transform(data['venue'])

data['winner'] = winner_encoder.transform(data['winner'])

# --------------------------------
# TRAIN MODEL
# --------------------------------

X = data.drop('winner', axis=1)

y = data['winner']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

# --------------------------------
# DROPDOWN VALUES
# --------------------------------

teams = sorted(team_encoder.classes_)

venues = sorted(venue_encoder.classes_)

# --------------------------------
# HOME PAGE
# --------------------------------

@app.route('/', methods=['GET', 'POST'])
def home():

    result = ""

    if request.method == 'POST':

        try:

            team1 = request.form['team1']

            team2 = request.form['team2']

            toss_winner = request.form['toss_winner']

            toss_decision = request.form['toss_decision']

            venue = request.form['venue']

            # Convert input into encoded values
            input_data = pd.DataFrame({

                'team1': [team_encoder.transform([team1])[0]],

                'team2': [team_encoder.transform([team2])[0]],

                'toss_winner': [team_encoder.transform([toss_winner])[0]],

                'toss_decision': [toss_encoder.transform([toss_decision])[0]],

                'venue': [venue_encoder.transform([venue])[0]]

            })

            # Predict
            prediction = model.predict(input_data)

            # Decode result
            winner = winner_encoder.inverse_transform(prediction)

            result = "Predicted Winner : " + winner[0]

        except Exception as e:

            result = "Error : " + str(e)

    return render_template(
        'index.html',
        result=result,
        teams=teams,
        venues=venues
    )

# --------------------------------
# RUN APP
# --------------------------------

if __name__ == '__main__':

    app.run(debug=True)