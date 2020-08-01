# Importing necessary libraries:

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
from modAL.utils.selection import weighted_random
from sklearn.model_selection import train_test_split
from modAL.disagreement import max_disagreement_sampling
from modAL.disagreement import vote_entropy_sampling
from modAL.disagreement import vote_entropy
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


# Splitting data into test and train

def split(dataset, train_size):
    x = dataset.values[:, 1:]
    y = dataset.values[:, 0]
    x_train, unlabel, y_train, label = train_test_split(x, y, train_size=train_size)
    # unlabel, x_test, label, y_test = train_test_split( x_pool, y_pool, test_size = test_size)
    return x_train, y_train, unlabel, label


# Defining random classifier

def random_sampling(classifier, X_pool, n_instances=1):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples), n_instances)
    return query_idx, X_pool[query_idx]


label_encoder = LabelEncoder()

raw_data = pd.read_csv("letter-recognition.data", sep=',')
raw_data['letter'] = label_encoder.fit_transform(raw_data['letter'])

x_raw, y_raw, x_unused, y_unused = split(raw_data, 0.1)
raw_data_array = np.concatenate((y_raw.reshape(-1, 1), x_raw), axis=1)
raw_data_used = pd.DataFrame(data=raw_data_array)

n_members = 6
learner_list0 = list()
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
    learner_list0.append(learner0)

# assembling the committee
committee0 = Committee(learner_list=learner_list0)
x_raw0, y_raw0 = x_raw, y_raw

x_train, y_train, x_pool, y_pool = split(raw_data_used, 0.1)

# Learner Definitions
learnerPool0 = ActiveLearner(estimator=RandomForestClassifier(), query_strategy=uncertainty_sampling,
                             X_training=x_train,
                             y_training=y_train)

learnerRandom0 = ActiveLearner(estimator=RandomForestClassifier(), query_strategy=random_sampling, X_training=x_train,
                               y_training=y_train)

learnerStream0 = ActiveLearner(estimator=RandomForestClassifier(), query_strategy=uncertainty_sampling,
                               X_training=x_train, y_training=y_train)
predictions = learnerStream0.predict(x_raw)
iS_correct = (predictions == y_raw)

unqueried_score0 = learnerPool0.score(x_unused, y_unused)
unqueried_score1 = learnerRandom0.score(x_unused, y_unused)
unqueried_score2 = learnerStream0.score(x_unused, y_unused)

performance_history = [
    [unqueried_score2, committee0.score(x_unused, y_unused), unqueried_score0, unqueried_score1]]

x_Stream_temp0, y_Stream_temp0 = x_pool, y_pool

x_pool_temp0, y_pool_temp0, x_pool_temp1, y_pool_temp1 = x_pool, y_pool, x_pool, y_pool

# Loop for Training on 20% data for all the 4 learners (1 stream, 1 pool, 1 committee, 1 random)
while len(x_pool_temp0) > 1200:
    query_index0, query_instance0 = learnerPool0.query(x_pool_temp0, n_instances=10)
    query_index1, query_instance1 = learnerRandom0.query(x_pool_temp1, n_instances=10)

    X0, y0 = x_pool_temp0[query_index0].reshape(10, -1), y_pool_temp0[query_index0].reshape(10, )
    X1, y1 = x_pool_temp1[query_index1].reshape(10, -1), y_pool_temp1[query_index1].reshape(10, )

    learnerPool0.teach(X=X0, y=y0)
    learnerRandom0.teach(X=X1, y=y1)

    x_pool_temp0, y_pool_temp0 = np.delete(x_pool_temp0, query_index0, axis=0), np.delete(y_pool_temp0, query_index0)
    x_pool_temp1, y_pool_temp1 = np.delete(x_pool_temp1, query_index1, axis=0), np.delete(y_pool_temp1, query_index1)

    model_accuracy0 = learnerPool0.score(x_unused, y_unused)
    model_accuracy1 = learnerRandom0.score(x_unused, y_unused)

    query_index2, query_instance2 = max_disagreement_sampling(committee0, x_raw0, n_instances=10)
    committee0.teach(X=x_raw0[query_index2].reshape(10, -1), y=y_raw0[query_index2].reshape(10, ))

    stream_idx0 = np.random.choice(range(len(x_Stream_temp0)), size=5)

    uncertainty_stream_0 = np.mean(
        [classifier_uncertainty(learnerStream0, x_Stream_temp0[i].reshape(1, -1)) for i in stream_idx0])
    if uncertainty_stream_0 >= 0.3:
        X0, y0 = x_Stream_temp0[stream_idx0].reshape(5, -1), y_Stream_temp0[stream_idx0].reshape(5, )
        learnerStream0.teach(X=X0, y=y0)
    new_score0 = learnerStream0.score(x_unused, y_unused)

    x_stream_temp0, y_stream_temp0 = np.delete(x_Stream_temp0, stream_idx0, axis=0), np.delete(y_Stream_temp0,
                                                                                               stream_idx0)

    x_raw0 = np.delete(x_raw0, query_index2, axis=0)
    y_raw0 = np.delete(y_raw0, query_index2)

    print(performance_history[-1])
    performance_history.append(
        [new_score0, committee0.score(x_unused, y_unused),
         model_accuracy0, model_accuracy1])

print(performance_history[-1])

# Plotting the graph:
performance_plot = [[], [], [], []]
for x in performance_history:
    performance_plot[0].append(x[0])
    performance_plot[1].append(x[1])
    performance_plot[2].append(x[2])
    performance_plot[3].append(x[3])
performance_plot = np.asarray(performance_plot)
plt.plot(performance_plot[0], label="Stream based Sampling")
plt.plot(performance_plot[1], label="Committee Sampling")
plt.plot(performance_plot[2], label="Pool based Sampling")
plt.plot(performance_plot[3], label="Random Sampling")
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Model comparison with different methods')
plt.legend()
plt.show()
