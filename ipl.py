import pandas as pd 
import numpy as np
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')
match.head()
match.shape
delivery.head()
delivery.groupby(['match_id' , 'inning']).sum()['total_runs']
total_score_df = delivery.groupby(['match_id' , 'inning']).sum()['total_runs'].reset_index()
total_score_df
total_score_df = total_score_df[total_score_df['inning'] == 1]
total_score_df
match_df = match.merge(total_score_df[['match_id' , 'total_runs' ]] , left_on = 'id' , right_on =  'match_id')

match_df
match_df['team1'].unique()
teams  = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bengaluru',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Lucknow Super Giants',
    'Gujarat Titans'
]
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

match_df['team1'] = match_df['team1'].str.replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
match_df['team2'] = match_df['team2'].str.replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')

match_df['team1'] = match_df['team1'].str.replace('Punjab Kings ', 'Kings XI Punjab')
match_df['team2'] = match_df['team2'].str.replace('Punjab Kings ', 'Kings XI Punjab ')

match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]
match_df.shape
match_df = match_df[['match_id' , 'city' , 'winner' , 'total_runs']]
match_df
delivery_df = match_df.merge(delivery , on = 'match_id')
delivery_df
delivery_df = delivery_df[delivery_df['inning'] == 2]
delivery_df.shape
delivery_df
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
delivery_df
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
delivery_df
delivery_df['balls_left'] = 120 - (delivery_df['over']*6 + delivery_df['ball'])
delivery_df
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')

wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values


delivery_df['wickets'] = 10 - wickets
delivery_df.head()
delivery_df
delivery_df.head()
# Current run rate = (current_score / balls)*6

delivery_df['crr'] = (delivery_df['current_score']*6)/(120-delivery_df['balls_left'])
# rrr. Required run rate
delivery_df['rrr'] = (delivery_df['runs_left'] * 6)/(delivery_df['balls_left'])
delivery_df
def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0
delivery_df['result'] = delivery_df.apply(result , axis = 1)
delivery_df
final_df = delivery_df[['batting_team' , 'bowling_team' , 'city' , 'runs_left' , 'balls_left' , 'wickets' , 'total_runs_x' , 'crr' , 'rrr' , 'result']]
final_df
final_df = final_df.sample(final_df.shape[0])
final_df.sample()
final_df.isnull().sum()
final_df.dropna(inplace=True)
final_df = final_df[final_df['balls_left'] != 0]
x = final_df.iloc[: , :-1]
y = final_df.iloc[: , -1]
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state = 1)
x_train
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf' , OneHotEncoder(sparse_output=False,drop='first') , ['batting_team' , 'bowling_team' , 'city'])
]
,remainder = 'passthrough')
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver = 'liblinear'))
])
pipe.fit(x_train , y_train)
#x_train.describe()
y_pred = pipe.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test , y_pred)
#pipe.predict_proba(x_test)[10]
def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))
def match_progression(x_df, match_id, pipe):
    match = x_df[x_df['match_id'] == match_id]
    
    if match.empty:
        print(f"No data found for match_id {match_id}")
        return None, None

    match = match[(match['ball'] == 6)]
    
    temp_df = match[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']].dropna()
    
    temp_df = temp_df[temp_df['balls_left'] != 0]
    
    if temp_df.empty:
        print(f"No data left after filtering for match_id {match_id}")
        return None, None
    
    result = pipe.predict_proba(temp_df)
    
    temp_df['lose'] = np.round(result.T[0] * 100, 1)
    temp_df['win'] = np.round(result.T[1] * 100, 1)
    
    temp_df['end_of_over'] = range(1, temp_df.shape[0] + 1)
    
    target = temp_df['total_runs_x'].values[0]
    
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0, target)  
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0, 10)  
    wickets.append(0)  
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[:temp_df.shape[0]]  
    
    print("Target:", target)
    
    temp_df = temp_df[['end_of_over', 'runs_after_over', 'wickets_in_over', 'lose', 'win']]
    return temp_df, target

temp_df, target = match_progression(delivery_df, 335990 , pipe)
temp_df
from matplotlib import pyplot as plt


plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))