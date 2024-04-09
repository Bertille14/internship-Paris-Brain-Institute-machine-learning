import numpy as np
import csv
import pandas as pd
import torch
import time
import string
import pickle
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from networkx.algorithms import community
from cdlib import algorithms
import community

filename = 'C:/Users/Bertille/Downloads/Stage ICM 2A/AUTOMATISATION AUT/distancesAUT2.csv'


# LOOP THROUGH DATASETS
start_time = time.time()  # keep track of speed

# GRAB CURRENT MODEL NAME
model = SentenceTransformer('dangvantuan/sentence-camembert-large')

# LOAD CURRENT DATA
d = pd.read_csv(filename, delimiter=';')

cue_embeddings = {}  # Dictionary to store embeddings for each cue
similarity_info = {}  # Dictionary to store similarity info for each cue

#BUILD SENTENCES
for index, row in d.iterrows():
    cue = str(row["cue"])  # Convert cue to string
    cue_lower=cue.lower()
    cue_upper=cue.upper()
    response = row["resp"]  # get current response
    if isinstance(response, str):
        response_split = response.split()
    else:
        response_split = []
    if cue not in response_split and cue_lower not in response_split and cue_upper not in response_split: #verifie que la cue est pas en majuscule ou minuscule dans la reponse, l'ajoute si absent
        sentence = cue + ' ' + str(response)
    else:
        sentence = response


    if cue not in cue_embeddings: #créé une liste de phrases pour chaque nouvelle cue
        cue_embeddings[cue] = []

# Ajoute la phrase avec son embedding à la liste de la cue
    embedding = model.encode(sentence)  # encode the sentence into an embedding
    cue_embeddings[cue].append((sentence, embedding))  # Utilize new_sentence here
    #print(sentence)

# Calculate similarity matrices for each cue
for cue, sentence_embeddings in cue_embeddings.items():
    sentences, embeddings = zip(*sentence_embeddings)
    embeddings = np.array(embeddings)

    # Calculate cosine similarity matrix using cosine_similarity
    sim = cosine_similarity(embeddings)

    num_phrases = len(sim)



    # Créer un graphe non orienté à partir de la matrice de distances
    G = nx.Graph()

    # Ajouter les nœuds au graphe
    G.add_nodes_from(range(1,num_phrases+1))
    node_to_phrase = {i + 1: sentences[i] for i in range(num_phrases)}

    # Ajouter les arêtes pondérées (en fonction des distances) entre les nœuds
    for i in range(num_phrases):
        for j in range(i + 1, num_phrases):
            weight = sim[i][j]  # Convertir la distance en similarité
            G.add_edge(i+1, j+1, weight=weight) #numérotation de 1 à nombre de phrases
    # Utiliser l'algorithme de Louvain pour la détection de communautés

    partition = community.best_partition(G, weight='weight')

    # Créer un dictionnaire pour stocker les communautés et leurs phrases associées
    communities = {}
    print(f"Cue:{cue}")
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)
        print(f"Phrase {node}: {node_to_phrase[node]} in Community {community_id}")



#Plot heatmap for each cue
for cue, similarity_info in similarity_info.items():
    similarity_matrix = similarity_info['similarity_matrix']
    sentences = similarity_info['sentences']

    plt.figure(figsize=(10, 8))
    plt.title(f"Heatmap for Cue: {cue}")
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.xticks(range(len(sentences)), sentences, rotation='vertical')
    plt.yticks(range(len(sentences)), sentences)
    plt.colorbar()
    plt.show()


    # # Dessiner le graphe avec les couleurs de communauté
    # pos = nx.spring_layout(G)
    # node_colors = [partition[node] for node in G.nodes()]
    # plt.figure(figsize=(10, 8))
    # nx.draw(G, pos, node_color=node_colors, with_labels=True)
    # plt.title(f"Cue: {cue}")
    # plt.show()
