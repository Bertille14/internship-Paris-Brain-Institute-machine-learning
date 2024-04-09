
import numpy as np
import csv
import pandas as pd
import torch
import time
import string
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim



filename= 'C:/Users/Bertille/Downloads/Stage ICM 2A/AUTOMATISATION AUT/distanceAUTtest.csv'

    # LOOP THROUGH DATASETS
start_time = time.time()  # keep track of speed

model = SentenceTransformer('bert-base-multilingual-cased')

d = pd.read_csv(filename, delimiter=';')

features_cue = []
features_resp = []
sentence=[]
for index, row in d.iterrows():
    ID = row["sub"]  # get current participant ID
    cue = row["cue"]  # get current cue
    response = row["resp"]  # get current response

    sentence.append(cue+" "+response) # combine the cue and response into a
    features_cue.append(cue)
    features_resp.append(response)

    np.save("features_resp_", features_resp)
    np.save("features_cue_", features_cue)
print(sentence)
embeddings = model.encode(sentence)
embeddings.shape


sim = np.zeros((len(sentences), len(sentence)))

for i in range(len(sentence)):
    sim[i:,i] = cos_sim(embeddings[i], embeddings[i:])


# print(sentences)
# print(f"Similarity matrix for cue '{cue_to_retrieve}':")
# print(similarity_matrix)
