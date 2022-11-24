from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

def prediction(p1, Players, n, users):
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