import pandas as pd
import numpy as np
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import uncertainty_sampling
from modAL.uncertainty import margin_sampling
from modAL.uncertainty import entropy_sampling
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from modAL.uncertainty import classifier_uncertainty
import matplotlib.pyplot as plt
from modAL.disagreement import max_disagreement_sampling
from modAL.disagreement import vote_entropy_sampling
from modAL.disagreement import vote_entropy
from modAL.disagreement import KL_max_disagreement
import pickle

# Using Previously Initialized Files
pkl_filename = "x_unused.pkl"
with open(pkl_filename, 'rb') as file:
    x_unused = pickle.load(file)
pkl_filename = "y_unused.pkl"
with open(pkl_filename, 'rb') as file:
    y_unused = pickle.load(file)
pkl_filename = "committee_KL_Max.pkl"
with open(pkl_filename, 'rb') as file:
    committee0 = pickle.load(file)

count = 0
idx = 0
list_idx = []
# Calculating Version space using Vote Entropy in the pool of 18000 elements in the committee
for x in x_unused:
    temp = vote_entropy(committee0, x.reshape(1, -1))
    print(count)
    if temp != 0:
        list_idx.append([temp, idx])
        count += 1
    idx += 1
# Sorting the list according to max entropy with indexes of elements
list_idx = sorted(list_idx, reverse=True)
print(count)
print(list_idx)
# Storing the List of elements with vote entropy and its index
pkl_filename = "list_version_space.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(list_idx, file)

# Version Space 2616
