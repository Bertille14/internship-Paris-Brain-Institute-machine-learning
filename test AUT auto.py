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


filename = 'C:/Users/Bertille/Downloads/Stage ICM 2A/AUTOMATISATION AUT/distanceAUTtest.csv'

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
    cue = row["cue"]  # get current cue
    cue_lower=cue.lower()
    cue_upper=cue.upper()
    response = row["resp"]  # get current response
    response_split = response.split()
    if cue not in response_split and cue_lower not in response_split and cue_upper not in response_split: #verifie que la cue est pas en majuscule ou minuscule dans la reponse, l'ajoute si absent
        sentence=cue+' '+response
    else:
        sentence=response

#BUILD EMBEDDINGS
    embedding = model.encode(sentence)  # encode the sentence into an embedding
    if cue not in cue_embeddings: #créé une liste de phrases pour chaque nouvelle cue
        cue_embeddings[cue] = []
    cue_embeddings[cue].append((sentence, embedding))  # Utilize new_sentence here
    #print(sentence)

# Calculate similarity matrices for each cue
for cue, sentence_embeddings in cue_embeddings.items():
    sentences, embeddings = zip(*sentence_embeddings)
    embeddings = np.array(embeddings)

    # Calculate cosine similarity matrix using cosine_similarity
    sim = cosine_similarity(embeddings)

    # Keep only the lower triangular part of the similarity matrix
    #sim = np.tril(sim)

    # Store similarity information in the dictionary
    similarity_info[cue] = {
        'sentences': sentences,
        'similarity_matrix': sim
    }

    # Save sentences and similarity matrix using np.save
    np.save(f"sentences_{cue}.npy", sentences)
    np.save(f"similarity_matrix_{cue}.npy", sim)

    #print(f"Cosine similarity matrix for cue '{cue}':")
    #print(sim[:5, :5])

# Save the similarity_info dictionary as a separate file


with open('similarity_info.pkl', 'wb') as file:
    pickle.dump(similarity_info, file)

# Load the similarity_info dictionary
with open('similarity_info.pkl', 'rb') as file:
    similarity_info = pickle.load(file)

# Create and visualize complete graphs
for cue, info in similarity_info.items():
    sentences = info['sentences']
    sim_matrix = info['similarity_matrix']

    num_sentences = len(sentences)

    # Create a complete graph
    Gcomplete = nx.complete_graph(num_sentences)

    # Add cosine similarity as edge weights for complete graphs
    for i in range(num_sentences):
        for j in range(i + 1, num_sentences):
            similarity = sim_matrix[i, j]
            Gcomplete[i][j]['weight'] = similarity

    # Visualize the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(Gcomplete)  # positions for all nodes
    labels = {i: sentences[i] for i in range(num_sentences)}
    nx.draw(Gcomplete, pos, labels=labels, with_labels=True, font_size=8, node_size=1000)
    edge_labels = nx.get_edge_attributes(Gcomplete, 'weight')
    nx.draw_networkx_edge_labels(Gcomplete, pos, edge_labels=edge_labels, font_size=8)
    plt.title(f"Complete Graph for '{cue}'")
    plt.show()

    # Create an incomplete graph
    Gincomplete=Gcomplete.copy()

    threshold = 0.4  # Choisir un seuil pour déterminer quelles arêtes supprimer
    edges_to_remove = [(u, v) for u, v, d in Gincomplete.edges(data=True) if d['weight'] < threshold]
    Gincomplete.remove_edges_from(edges_to_remove)

  # Add cosine similarity as edge weights for the incomplete graph
    for i in range(num_sentences):
        for j in range(i + 1, num_sentences):
            if (i, j) in Gincomplete.edges():
                Gincomplete[i][j]['weight'] = sim_matrix[i, j]


# Visualize the incomplete graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(Gincomplete)  # positions for all nodes
    labels = {i: sentences[i] for i in range(num_sentences)}
    nx.draw(Gincomplete, pos, labels=labels, with_labels=True, font_size=8, node_size=1000)
    edge_labels = nx.get_edge_attributes(Gincomplete, 'weight')
    nx.draw_networkx_edge_labels(Gincomplete, pos, edge_labels=edge_labels, font_size=8)
    plt.title(f"Incomplete Graph for '{cue}'")
    plt.show()

    # coms_complete = algorithms.louvain(Gcomplete, weight='weight', resolution=1., randomize=False)
    # pos = nx.spring_layout(Gcomplete)
    # colors_complete = [coms_complete.communities[node] for node in Gcomplete.nodes()]
    #
    # nx.draw(Gcomplete, pos, node_color=colors_complete, with_labels=True)
    # plt.title(f"Louvain Communities in Complete Graph for '{cue}'")
    # plt.show()
    #
    # # Create and visualize the Louvain communities in the incomplete graph
    # coms_incomplete = algorithms.louvain(Gincomplete, weight='weight', resolution=1., randomize=False)
    # pos = nx.spring_layout(Gincomplete)
    # colors_incomplete = [coms_incomplete.communities[node] for node in Gincomplete.nodes()]
    #
    # nx.draw(Gincomplete, pos, node_color=colors_incomplete, with_labels=True)
    # plt.title(f"Louvain Communities in Incomplete Graph for '{cue}'")
    # plt.show()


# Remove edge weights (make it unweighted)
    for u, v in Gincomplete.edges():
        del Gincomplete[u][v]['weight']

# Visualize the unweighted graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(Gincomplete)  # positions for all nodes
    labels = {i: sentences[i] for i in range(num_sentences)}
    nx.draw(Gincomplete, pos, labels=labels, with_labels=True, font_size=8, node_size=1000)
    plt.title(f"Unweighted Graph for '{cue}'")
    plt.show()
#
#     # Appliquer l'algorithme Louvain à Gcomplete pour détecter les communautés
#     coms_complete = algorithms.louvain(Gcomplete, weight='weight', resolution=1., randomize=False)
#
#     # Visualiser les communautés détectées dans le graphe complet
#     pos = nx.spring_layout(Gcomplete)  # Définir la disposition des nœuds
#     colors = [coms_complete.communities[node] for node in Gcomplete.nodes()]  # Obtenir les couleurs des communautés
#
#     nx.draw(Gcomplete, pos, node_color=colors, with_labels=True)
#     plt.show()
#
#     # Appliquer l'algorithme Louvain à Gincomplete pour détecter les communautés
#     coms_incomplete = algorithms.louvain(Gincomplete, weight='weight', resolution=1., randomize=False)
#
#     # Visualiser les communautés détectées dans le graphe incomplet
#     pos = nx.spring_layout(Gincomplete)  # Définir la disposition des nœuds
#     colors = [coms_incomplete.communities[node] for node in Gincomplete.nodes()]  # Obtenir les couleurs des communautés
#
#     nx.draw(Gincomplete, pos, node_color=colors, with_labels=True)
#     plt.show()



# #Appliquer l'algorithme MCL sur la matrice de similarité
# coms = algorithms.markov_clustering(sim_matrix)
#
# #Afficher les communautés détectées
# for community in coms.communities:
#     print(community)

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