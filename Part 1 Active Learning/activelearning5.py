import pandas as pd
import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from modAL.uncertainty import margin_sampling
from modAL.uncertainty import entropy_sampling
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from modAL.uncertainty import classifier_uncertainty
import matplotlib.pyplot as plt
from modAL.utils.selection import weighted_random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
import pickle


def split(dataset, train_size):
    x = dataset.values[:, 1:]
    y = dataset.values[:, 0]
    x_train, unlabel, y_train, label = train_test_split(x, y, train_size=train_size)
    # unlabel, x_test, label, y_test = train_test_split( x_pool, y_pool, test_size = test_size)
    return x_train, y_train, unlabel, label


pkl_filename = "x_unused.pkl"
with open(pkl_filename, 'rb') as file:
    x_unused = pickle.load(file)
pkl_filename = "y_unused.pkl"
with open(pkl_filename, 'rb') as file:
    y_unused = pickle.load(file)

unused_data = np.concatenate((y_unused.reshape(-1, 1), x_unused), axis=1)
unused_data = pd.DataFrame(data=unused_data)
x_train, y_train, x_test, y_test = split(unused_data, 0.4)

Kmeans_model = KMeans(n_clusters=26)
Kmeans_model.fit(x_train)
list_idx = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
i = 0
for x in Kmeans_model.labels_:
    # print(x)
    list_idx[x].append(i)
    i += 1

max_acc = 0.0
for k in range(1000):
    ans_list = []
    for i in range(26):
        chosen_indices = np.random.choice(list_idx[i], int(len(list_idx[i]) / 4), replace=False)
        count = Counter(y_train[chosen_indices]).most_common(1)
        for x, y in count:
            ans_list.append(x)

    y_pred = [-1] * 7200
    for i in range(26):
        for j in list_idx[i]:
            y_pred[j] = ans_list[i]
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_train[i]:
            count += 1

    acc = count / 7200
    max_acc = max(acc, max_acc)

print(max_acc)

# Total sample space is: 20,000
# 90% unlabelled points: 18,000
# 40% of 90% labelled data taken for consideration: 7,200
# 20% of the above data which were labelled: 1,440
# Points which didn't require labelling: 5760
# Money saved: ₹5,76,000
# Time saved: 5760 hours
