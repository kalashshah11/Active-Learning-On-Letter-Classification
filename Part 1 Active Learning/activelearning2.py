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

# Define our PCA transformer and fit it onto our raw dataset.


from sklearn.model_selection import train_test_split


# RANDOM_STATE_SEED = 123
# np.random.seed(RANDOM_STATE_SEED)

def split(dataset, train_size):
    x = dataset.values[:, 1:]
    y = dataset.values[:, 0]
    x_train, unlabel, y_train, label = train_test_split(x, y, train_size=train_size)
    # unlabel, x_test, label, y_test = train_test_split( x_pool, y_pool, test_size = test_size)
    return x_train, y_train, unlabel, label


label_encoder = LabelEncoder()

raw_data = pd.read_csv("letter-recognition.data", sep=',')
raw_data['letter'] = label_encoder.fit_transform(raw_data['letter'])

x_raw, y_raw, x_unused, y_unused = split(raw_data, 0.1)
raw_data_array = np.concatenate((y_raw.reshape(-1, 1), x_raw), axis=1)
raw_data_used = pd.DataFrame(data=raw_data_array)

# print(raw_data_used)


n_members = 6
learner_list0 = list()
learner_list1 = list()
# Initialising Learner and adding it to the learner list
for member_idx in range(n_members):
    n_initial = 50
    train_idx = np.random.choice(range(x_raw.shape[0]), size=n_initial, replace=False)
    x_train = x_raw[train_idx]
    y_train = y_raw[train_idx]
    # creating a reduced copy of the data with the known instances removed
    x_raw = np.delete(x_raw, train_idx, axis=0)
    y_raw = np.delete(y_raw, train_idx)
    # initializing learner
    learner0 = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=x_train, y_training=y_train
    )
    learner1 = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=x_train, y_training=y_train
    )
    learner_list0.append(learner0)
    learner_list1.append(learner1)
    
    
# assembling the committee
committee0 = Committee(learner_list=learner_list0)
committee1 = Committee(learner_list=learner_list1)
#Initial Scores
unqueried_score = [committee0.score(x_unused, y_unused), committee1.score(x_unused, y_unused)]
print(unqueried_score)
x_raw0, y_raw0, x_raw1, y_raw1 = x_raw, y_raw, x_raw, y_raw

performance_history = [unqueried_score]
# query by committee
n_queries = 150
for idx in range(n_queries):  # chaiyegg
    query_idx0, query_instance0 = max_disagreement_sampling(committee0, x_raw0, n_instances=10)
    query_idx1, query_instance1 = vote_entropy_sampling(committee1, x_raw1, n_instances=10)
    # print(query_idx0, query_idx1)
    committee0.teach(X=x_raw0[query_idx0].reshape(10, -1), y=y_raw0[query_idx0].reshape(10, ))
    committee1.teach(X=x_raw1[query_idx1].reshape(10, -1), y=y_raw1[query_idx1].reshape(10, ))
    performance_history.append([committee0.score(x_unused, y_unused), committee1.score(x_unused, y_unused)])
    x_raw0 = np.delete(x_raw0, query_idx0, axis=0)
    y_raw0 = np.delete(y_raw0, query_idx0)
    x_raw1 = np.delete(x_raw1, query_idx1, axis=0)
    y_raw1 = np.delete(y_raw1, query_idx1)
    print(performance_history[-1])

print(vote_entropy(committee1, x_unused), KL_max_disagreement(committee0, x_unused))

performance_plot = [[], []]
for x in performance_history:
    performance_plot[0].append(x[0])
    performance_plot[1].append(x[1])

#Plotting the two measures of disagreement
performance_plot = np.asarray(performance_plot)
plt.plot(performance_plot[0], label="Vote Entropy")
plt.plot(performance_plot[1], label="KL MAX Disagreement")
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Disagreement in the measures')  # Ye name change kar le Mere s=dimaag mai kucch nahi aa raha
plt.legend()
plt.show()

#Storing files for later use
import pickle

pkl_filename = "committee_vote_entropy.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(committee1, file)

pkl_filename = "committee_KL_Max.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(committee0, file)

# DONE

