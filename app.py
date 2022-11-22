import gradio as gr
from joblib import load
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

solo = load('data/solo.joblib')
duo = load('data/duo.joblib')
trio = load('data/trio.joblib')
squad = load('data/squad.joblib')
users = load('data/users.joblib')

def plotOnMap(p1, Players, userList):
    pca = PCA(n_components = 2)
    playersPCA = pca.fit_transform(Players)
    p1PCA = pca.transform([p1])
    userListPCA = pca.transform(userList)
    
    fig = plt.figure()
    
    plt.scatter(playersPCA[:, 0], playersPCA[:, 1])
    plt.scatter(userListPCA[:, 0], userListPCA[:, 1], c = 'g')
    plt.scatter(p1PCA[:, 0], p1PCA[:, 1], c = 'r')

    plt.title("Players points")
    plt.ylabel("X2")
    plt.xlabel("X1")
    return fig

def prediction(p1, Players, n):
    df_players = pd.DataFrame(data=Players, columns=["score", "top1", "kd", "winRatio", "matches", "kills", "minutes"])

    Distances = []
    for p2 in Players:
        Distances.append((1 - spatial.distance.cosine(p1, p2))*100)
    
    df_players.insert(1, "Matching %", Distances)
    df_players = df_players.sort_values(by=['Matching %'], ascending=False)

    userIndexes = df_players.head(n).index
    userList = users[userIndexes]
    matchingPlayers = Players[userIndexes]

    fig = plotOnMap(p1,Players, matchingPlayers)

    return userList, fig

def printPlayers(players):
    label = "Matching players:\n"
    for player in players:
        label += f"\n{player[0]}"
    return label

def greet(type, score, top1, kd, winRatio, matches, kills, minutes):
    p1 = np.array([score, top1, kd, winRatio, matches, kills, minutes])

    if type == 'Solo': userList, fig = prediction(p1, solo, 1)
    if type == 'Duo': userList, fig = prediction(p1, duo, 3)
    if type == 'Trio': userList, fig =  prediction(p1, trio, 5)
    if type == 'Squad': userList, fig =  prediction(p1, squad, 7)

    userList = printPlayers(userList)

    return userList, fig

demo = gr.Interface(
    fn=greet,
    examples = [
        ["Duo", 0.730117966,0.735317356,0.763649425,0.81316726,0.264922696,0.496196684,0.492858304],
    ],
    inputs=[
        gr.Dropdown(["Solo", "Duo", "Trio", "Squad"]),
        "number", 
        "number", 
        "number", 
        "number", 
        "number", 
        "number", 
        "number"
    ],
    outputs=[gr.Text(label= "Similar"), gr.Plot()]
)

demo.launch(share=True)