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
import time
from modAL.utils.selection import weighted_random
from sklearn.model_selection import train_test_split


# RANDOM_STATE_SEED = 123
# np.random.seed(RANDOM_STATE_SEED)g


def split(dataset, train_size): # define custom method to split data
    x = dataset.values[:, 1:]
    y = dataset.values[:, 0]
    x_train, unlabel, y_train, label = train_test_split(x, y, train_size=train_size)
    # unlabel, x_test, label, ytest = train_test_split( x_pool, y_pool, test_size = test_size)
    return x_train, y_train, unlabel, label


label_encoder = LabelEncoder() # Label Encoder For converting the class to digits

raw_data = pd.read_csv("letter-recognition.data", sep=',')
raw_data['letter'] = label_encoder.fit_transform(raw_data['letter'])

x_raw, y_raw, x_unused, y_unused = split(raw_data, 0.1)
raw_data_array = np.concatenate((y_raw.reshape(-1, 1), x_raw), axis=1)
raw_data_used = pd.DataFrame(data=raw_data_array)

x_train, y_train, x_pool, y_pool = split(raw_data_used, 0.1) # Splitting data for initial training of 10% to the classifiers
y_trainEncoded = label_encoder.fit_transform(y_train)

# 24 Learners

# Learner Definations
learnerpool0 = ActiveLearner(estimator=RandomForestClassifier(), query_strategy=uncertainty_sampling,
                             X_training=x_train,
                             y_training=y_train)
learnerpool1 = ActiveLearner(estimator=RandomForestClassifier(), query_strategy=entropy_sampling, X_training=x_train,
                             y_training=y_train)
learnerpool2 = ActiveLearner(estimator=RandomForestClassifier(), query_strategy=margin_sampling, X_training=x_train,
                             y_training=y_train)
predictions = learnerpool0.predict(x_raw) 
is_correct = (predictions == y_raw)
unqueried_score0 = learnerpool0.score(x_unused, y_unused)
unqueried_score1 = learnerpool1.score(x_unused, y_unused)
unqueried_score2 = learnerpool2.score(x_unused, y_unused)

performance_history = [
    [unqueried_score0, unqueried_score1, unqueried_score2]]
# Allow our model to query our unlabeled dataset for the most
# informative points according to our query strategy (uncertainty sampling).

x_pool_temp0, y_pool_temp0, x_pool_temp1, y_pool_temp1, x_pool_temp2, y_pool_temp2 = x_pool, y_pool, x_pool, y_pool, x_pool, y_pool

# Loop for Pool Based Sampling 10%
while len(x_pool_temp0) > 1600:
    query_index0, query_instance0 = learnerpool0.query(x_pool_temp0, n_instances=10)
    query_index1, query_instance1 = learnerpool1.query(x_pool_temp1, n_instances=10)
    query_index2, query_instance2 = learnerpool2.query(x_pool_temp2, n_instances=10)

    # Teach our ActiveLearner model the record it has requested.

    X0, y0 = x_pool_temp0[query_index0].reshape(10, -1), y_pool_temp0[query_index0].reshape(10, )
    X1, y1 = x_pool_temp1[query_index1].reshape(10, -1), y_pool_temp1[query_index1].reshape(10, )
    X2, y2 = x_pool_temp2[query_index2].reshape(10, -1), y_pool_temp2[query_index2].reshape(10, )

    learnerpool0.teach(X=X0, y=y0)
    learnerpool1.teach(X=X1, y=y1)
    learnerpool2.teach(X=X2, y=y2)

    x_pool_temp0, y_pool_temp0 = np.delete(x_pool_temp0, query_index0, axis=0), np.delete(y_pool_temp0, query_index0)
    x_pool_temp1, y_pool_temp1 = np.delete(x_pool_temp1, query_index1, axis=0), np.delete(y_pool_temp1, query_index1)
    x_pool_temp2, y_pool_temp2 = np.delete(x_pool_temp2, query_index2, axis=0), np.delete(y_pool_temp2, query_index2)

    print(performance_history[-1])
    # Calculate and report our model's accuracy
    model_accuracy0 = learnerpool0.score(x_unused, y_unused)
    model_accuracy1 = learnerpool1.score(x_unused, y_unused)
    model_accuracy2 = learnerpool2.score(x_unused, y_unused)

    # Save our model's performance for plotting.
    performance_history.append([model_accuracy0, model_accuracy1, model_accuracy2])

print(performance_history[-1])
performance_plot = [[], [], []]
for x in performance_history:
    performance_plot[0].append(x[0])
    performance_plot[1].append(x[1])
    performance_plot[2].append(x[2])

# PLotting the 10% Pool Based Results
performance_plot = np.asarray(performance_plot)
plt.plot(performance_plot[0], label="Uncertainity Sampling")
plt.plot(performance_plot[1], label="Entropy Sampling")
plt.plot(performance_plot[2], label="Margin Sampling")
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('10% labelling Pool Based')
plt.legend()
plt.show()

x_pool_temp0, y_pool_temp0, x_pool_temp1, y_pool_temp1, x_pool_temp2, y_pool_temp2 = x_pool, y_pool, x_pool, y_pool, x_pool, y_pool

while len(x_pool_temp0) > 1400:
    query_index0, query_instance0 = learnerpool0.query(x_pool_temp0, n_instances=10)
    query_index1, query_instance1 = learnerpool1.query(x_pool_temp1, n_instances=10)
    query_index2, query_instance2 = learnerpool2.query(x_pool_temp2, n_instances=10)

    # Teach our ActiveLearner model the record it has requested.

    X0, y0 = x_pool_temp0[query_index0].reshape(10, -1), y_pool_temp0[query_index0].reshape(10, )
    X1, y1 = x_pool_temp1[query_index1].reshape(10, -1), y_pool_temp1[query_index1].reshape(10, )
    X2, y2 = x_pool_temp2[query_index2].reshape(10, -1), y_pool_temp2[query_index2].reshape(10, )

    learnerpool0.teach(X=X0, y=y0)
    learnerpool1.teach(X=X1, y=y1)
    learnerpool2.teach(X=X2, y=y2)

    x_pool_temp0, y_pool_temp0 = np.delete(x_pool_temp0, query_index0, axis=0), np.delete(y_pool_temp0, query_index0)
    x_pool_temp1, y_pool_temp1 = np.delete(x_pool_temp1, query_index1, axis=0), np.delete(y_pool_temp1, query_index1)
    x_pool_temp2, y_pool_temp2 = np.delete(x_pool_temp2, query_index2, axis=0), np.delete(y_pool_temp2, query_index2)

    print(performance_history[-1])
    # Calculate and report our model's accuracy
    model_accuracy0 = learnerpool0.score(x_unused, y_unused)
    model_accuracy1 = learnerpool1.score(x_unused, y_unused)
    model_accuracy2 = learnerpool2.score(x_unused, y_unused)

    # Save our model's performance for plotting.
    performance_history.append([model_accuracy0, model_accuracy1, model_accuracy2])

print(performance_history[-1])

performance_plot = [[], [], []]
for x in performance_history:
    performance_plot[0].append(x[0])
    performance_plot[1].append(x[1])
    performance_plot[2].append(x[2])

# PLotting the 20% Pool Based Results
performance_plot = np.asarray(performance_plot)
plt.plot(performance_plot[0], label="Uncertainity Sampling")
plt.plot(performance_plot[1], label="Entropy Sampling")
plt.plot(performance_plot[2], label="Margin Sampling")
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('20% labelling Pool Based')
plt.legend()
plt.show()

x_pool_temp0, y_pool_temp0, x_pool_temp1, y_pool_temp1, x_pool_temp2, y_pool_temp2 = x_pool, y_pool, x_pool, y_pool, x_pool, y_pool

while len(x_pool_temp0) > 1200:
    query_index0, query_instance0 = learnerpool0.query(x_pool_temp0, n_instances=10)
    query_index1, query_instance1 = learnerpool1.query(x_pool_temp1, n_instances=10)
    query_index2, query_instance2 = learnerpool2.query(x_pool_temp2, n_instances=10)

    # Teach our ActiveLearner model the record it has requested.

    X0, y0 = x_pool_temp0[query_index0].reshape(10, -1), y_pool_temp0[query_index0].reshape(10, )
    X1, y1 = x_pool_temp1[query_index1].reshape(10, -1), y_pool_temp1[query_index1].reshape(10, )
    X2, y2 = x_pool_temp2[query_index2].reshape(10, -1), y_pool_temp2[query_index2].reshape(10, )

    learnerpool0.teach(X=X0, y=y0)
    learnerpool1.teach(X=X1, y=y1)
    learnerpool2.teach(X=X2, y=y2)

    x_pool_temp0, y_pool_temp0 = np.delete(x_pool_temp0, query_index0, axis=0), np.delete(y_pool_temp0, query_index0)
    x_pool_temp1, y_pool_temp1 = np.delete(x_pool_temp1, query_index1, axis=0), np.delete(y_pool_temp1, query_index1)
    x_pool_temp2, y_pool_temp2 = np.delete(x_pool_temp2, query_index2, axis=0), np.delete(y_pool_temp2, query_index2)

    print(performance_history[-1])
    # Calculate and report our model's accuracy
    model_accuracy0 = learnerpool0.score(x_unused, y_unused)
    model_accuracy1 = learnerpool1.score(x_unused, y_unused)
    model_accuracy2 = learnerpool2.score(x_unused, y_unused)

    # Save our model's performance for plotting.
    performance_history.append([model_accuracy0, model_accuracy1, model_accuracy2])

print(performance_history[-1])
performance_plot = [[], [], []]
for x in performance_history:
    performance_plot[0].append(x[0])
    performance_plot[1].append(x[1])
    performance_plot[2].append(x[2])

# PLotting the 30% Pool Based Results
performance_plot = np.asarray(performance_plot)
plt.plot(performance_plot[0], label="Uncertainity Sampling")
plt.plot(performance_plot[1], label="Entropy Sampling")
plt.plot(performance_plot[2], label="Margin Sampling")
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('30% labelling Pool Based')
plt.legend()
plt.show()

x_pool_temp0, y_pool_temp0, x_pool_temp1, y_pool_temp1, x_pool_temp2, y_pool_temp2 = x_pool, y_pool, x_pool, y_pool, x_pool, y_pool

while len(x_pool_temp0) > 1000:
    query_index0, query_instance0 = learnerpool0.query(x_pool_temp0, n_instances=10)
    query_index1, query_instance1 = learnerpool1.query(x_pool_temp1, n_instances=10)
    query_index2, query_instance2 = learnerpool2.query(x_pool_temp2, n_instances=10)

    # Teach our ActiveLearner model the record it has requested.

    X0, y0 = x_pool_temp0[query_index0].reshape(10, -1), y_pool_temp0[query_index0].reshape(10, )
    X1, y1 = x_pool_temp1[query_index1].reshape(10, -1), y_pool_temp1[query_index1].reshape(10, )
    X2, y2 = x_pool_temp2[query_index2].reshape(10, -1), y_pool_temp2[query_index2].reshape(10, )

    learnerpool0.teach(X=X0, y=y0)
    learnerpool1.teach(X=X1, y=y1)
    learnerpool2.teach(X=X2, y=y2)

    x_pool_temp0, y_pool_temp0 = np.delete(x_pool_temp0, query_index0, axis=0), np.delete(y_pool_temp0, query_index0)
    x_pool_temp1, y_pool_temp1 = np.delete(x_pool_temp1, query_index1, axis=0), np.delete(y_pool_temp1, query_index1)
    x_pool_temp2, y_pool_temp2 = np.delete(x_pool_temp2, query_index2, axis=0), np.delete(y_pool_temp2, query_index2)

    print(performance_history[-1])
    # Calculate and report our model's accuracy
    model_accuracy0 = learnerpool0.score(x_unused, y_unused)
    model_accuracy1 = learnerpool1.score(x_unused, y_unused)
    model_accuracy2 = learnerpool2.score(x_unused, y_unused)

    # Save our model's performance for plotting.
    performance_history.append([model_accuracy0, model_accuracy1, model_accuracy2])

print(performance_history[-1])
performance_plot = [[], [], []]
for x in performance_history:
    performance_plot[0].append(x[0])
    performance_plot[1].append(x[1])
    performance_plot[2].append(x[2])

# PLotting the 40% Pool Based Results
performance_plot = np.asarray(performance_plot)
plt.plot(performance_plot[0], label="Uncertainity Sampling")
plt.plot(performance_plot[1], label="Entropy Sampling")
plt.plot(performance_plot[2], label="Margin Sampling")
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('40% labelling Pool Based')
plt.legend()
plt.show()

# Its Time to Stream Based Pooling


# Learner Definitions

learnerstream0 = ActiveLearner(estimator=RandomForestClassifier(), query_strategy=uncertainty_sampling,
                               X_training=x_train, y_training=y_train)
learnerstream1 = ActiveLearner(estimator=RandomForestClassifier(), query_strategy=entropy_sampling, X_training=x_train,
                               y_training=y_train)
learnerstream2 = ActiveLearner(estimator=RandomForestClassifier(), query_strategy=margin_sampling, X_training=x_train,
                               y_training=y_train)
predictions = learnerstream0.predict(x_raw)
is_correct = (predictions == y_raw)
unqueried_score0 = learnerstream0.score(x_unused, y_unused)
unqueried_score1 = learnerstream1.score(x_unused, y_unused)
unqueried_score2 = learnerstream2.score(x_unused, y_unused)

performance_history_stream = [[unqueried_score0, unqueried_score1, unqueried_score2]]

x_stream_temp0, y_stream_temp0, x_stream_temp1, y_stream_temp1, x_stream_temp2, y_stream_temp2 = x_pool, y_pool, x_pool, y_pool, x_pool, y_pool
while len(x_stream_temp0) > 1600:

    stream_idx0 = np.random.choice(range(len(x_stream_temp0)), size=5)
    stream_idx1 = np.random.choice(range(len(x_stream_temp1)), size=5)
    stream_idx2 = np.random.choice(range(len(x_stream_temp2)), size=5)

    uncertainty_stream_0 = np.mean(
        [classifier_uncertainty(learnerstream0, x_stream_temp0[i].reshape(1, -1)) for i in stream_idx0])
    if uncertainty_stream_0 >= 0.3:
        X0, y0 = x_stream_temp0[stream_idx0].reshape(5, -1), y_stream_temp0[stream_idx0].reshape(5, )
        learnerstream0.teach(X=X0, y=y0)
    new_score0 = learnerstream0.score(x_unused, y_unused)
    uncertainty_stream_1 = np.mean(
        [classifier_uncertainty(learnerstream1, x_stream_temp1[i].reshape(1, -1)) for i in stream_idx1])
    if uncertainty_stream_1 >= 0.3:
        X1, y1 = x_stream_temp1[stream_idx1].reshape(5, -1), y_stream_temp1[stream_idx1].reshape(5, )
        learnerstream1.teach(X=X1, y=y1)
    new_score1 = learnerstream1.score(x_unused, y_unused)
    uncertainty_stream_2 = np.mean(
        [classifier_uncertainty(learnerstream2, x_stream_temp2[i].reshape(1, -1)) for i in stream_idx2])
    if uncertainty_stream_2 >= 0.3:
        X2, y2 = x_stream_temp2[stream_idx2].reshape(5, -1), y_stream_temp2[stream_idx2].reshape(5, )
        learnerstream2.teach(X=X2, y=y2)
    new_score2 = learnerstream2.score(x_unused, y_unused)

    performance_history_stream.append([new_score0, new_score1, new_score2])
    x_stream_temp0, y_stream_temp0 = np.delete(x_stream_temp0, stream_idx0, axis=0), np.delete(y_stream_temp0,
                                                                                               stream_idx0)
    x_stream_temp1, y_stream_temp1 = np.delete(x_stream_temp1, stream_idx1, axis=0), np.delete(y_stream_temp1,
                                                                                               stream_idx1)
    x_stream_temp2, y_stream_temp2 = np.delete(x_stream_temp2, stream_idx2, axis=0), np.delete(y_stream_temp2,
                                                                                               stream_idx2)
    print(performance_history_stream[-1])

performance_plot = [[], [], []]
for x in performance_history_stream:
    performance_plot[0].append(x[0])
    performance_plot[1].append(x[1])
    performance_plot[2].append(x[2])

# PLotting the 10% Stream Based Results
performance_plot = np.asarray(performance_plot)
plt.plot(performance_plot[0], label="Uncertainity Sampling")
plt.plot(performance_plot[1], label="Entropy Sampling")
plt.plot(performance_plot[2], label="Margin Sampling")
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('10% labelling Stream Based')
plt.legend()
plt.show()
while len(x_stream_temp0) > 1400:

    stream_idx0 = np.random.choice(range(len(x_stream_temp0)), size=5)
    stream_idx1 = np.random.choice(range(len(x_stream_temp1)), size=5)
    stream_idx2 = np.random.choice(range(len(x_stream_temp2)), size=5)

    uncertainty_stream_0 = np.mean(
        [classifier_uncertainty(learnerstream0, x_stream_temp0[i].reshape(1, -1)) for i in stream_idx0])
    if uncertainty_stream_0 >= 0.3:
        X0, y0 = x_stream_temp0[stream_idx0].reshape(5, -1), y_stream_temp0[stream_idx0].reshape(5, )
        learnerstream0.teach(X=X0, y=y0)
    new_score0 = learnerstream0.score(x_unused, y_unused)
    uncertainty_stream_1 = np.mean(
        [classifier_uncertainty(learnerstream1, x_stream_temp1[i].reshape(1, -1)) for i in stream_idx1])
    if uncertainty_stream_1 >= 0.3:
        X1, y1 = x_stream_temp1[stream_idx1].reshape(5, -1), y_stream_temp1[stream_idx1].reshape(5, )
        learnerstream1.teach(X=X1, y=y1)
    new_score1 = learnerstream1.score(x_unused, y_unused)
    uncertainty_stream_2 = np.mean(
        [classifier_uncertainty(learnerstream2, x_stream_temp2[i].reshape(1, -1)) for i in stream_idx2])
    if uncertainty_stream_2 >= 0.3:
        X2, y2 = x_stream_temp2[stream_idx2].reshape(5, -1), y_stream_temp2[stream_idx2].reshape(5, )
        learnerstream2.teach(X=X2, y=y2)
    new_score2 = learnerstream2.score(x_unused, y_unused)

    performance_history_stream.append([new_score0, new_score1, new_score2])
    x_stream_temp0, y_stream_temp0 = np.delete(x_stream_temp0, stream_idx0, axis=0), np.delete(y_stream_temp0,
                                                                                               stream_idx0)
    x_stream_temp1, y_stream_temp1 = np.delete(x_stream_temp1, stream_idx1, axis=0), np.delete(y_stream_temp1,
                                                                                               stream_idx1)
    x_stream_temp2, y_stream_temp2 = np.delete(x_stream_temp2, stream_idx2, axis=0), np.delete(y_stream_temp2,
                                                                                               stream_idx2)
    print(performance_history_stream[-1])

performance_plot = [[], [], []]
for x in performance_history_stream:
    performance_plot[0].append(x[0])
    performance_plot[1].append(x[1])
    performance_plot[2].append(x[2])

# PLotting the 20% Stream Based Results
performance_plot = np.asarray(performance_plot)
plt.plot(performance_plot[0], label="Uncertainity Sampling")
plt.plot(performance_plot[1], label="Entropy Sampling")
plt.plot(performance_plot[2], label="Margin Sampling")
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('20% labelling Stream Based')
plt.legend()
plt.show()

while len(x_stream_temp0) > 1200:

    stream_idx0 = np.random.choice(range(len(x_stream_temp0)), size=5)
    stream_idx1 = np.random.choice(range(len(x_stream_temp1)), size=5)
    stream_idx2 = np.random.choice(range(len(x_stream_temp2)), size=5)

    uncertainty_stream_0 = np.mean(
        [classifier_uncertainty(learnerstream0, x_stream_temp0[i].reshape(1, -1)) for i in stream_idx0])
    if uncertainty_stream_0 >= 0.3:
        X0, y0 = x_stream_temp0[stream_idx0].reshape(5, -1), y_stream_temp0[stream_idx0].reshape(5, )
        learnerstream0.teach(X=X0, y=y0)
    new_score0 = learnerstream0.score(x_unused, y_unused)
    uncertainty_stream_1 = np.mean(
        [classifier_uncertainty(learnerstream1, x_stream_temp1[i].reshape(1, -1)) for i in stream_idx1])
    if uncertainty_stream_1 >= 0.3:
        X1, y1 = x_stream_temp1[stream_idx1].reshape(5, -1), y_stream_temp1[stream_idx1].reshape(5, )
        learnerstream1.teach(X=X1, y=y1)
    new_score1 = learnerstream1.score(x_unused, y_unused)
    uncertainty_stream_2 = np.mean(
        [classifier_uncertainty(learnerstream2, x_stream_temp2[i].reshape(1, -1)) for i in stream_idx2])
    if uncertainty_stream_2 >= 0.3:
        X2, y2 = x_stream_temp2[stream_idx2].reshape(5, -1), y_stream_temp2[stream_idx2].reshape(5, )
        learnerstream2.teach(X=X2, y=y2)
    new_score2 = learnerstream2.score(x_unused, y_unused)

    performance_history_stream.append([new_score0, new_score1, new_score2])
    x_stream_temp0, y_stream_temp0 = np.delete(x_stream_temp0, stream_idx0, axis=0), np.delete(y_stream_temp0,
                                                                                               stream_idx0)
    x_stream_temp1, y_stream_temp1 = np.delete(x_stream_temp1, stream_idx1, axis=0), np.delete(y_stream_temp1,
                                                                                               stream_idx1)
    x_stream_temp2, y_stream_temp2 = np.delete(x_stream_temp2, stream_idx2, axis=0), np.delete(y_stream_temp2,
                                                                                               stream_idx2)
    print(performance_history_stream[-1])

performance_plot = [[], [], []]
for x in performance_history_stream:
    performance_plot[0].append(x[0])
    performance_plot[1].append(x[1])
    performance_plot[2].append(x[2])

# PLotting the 30% Stream Based Results
performance_plot = np.asarray(performance_plot)
plt.plot(performance_plot[0], label="Uncertainity Sampling")
plt.plot(performance_plot[1], label="Entropy Sampling")
plt.plot(performance_plot[2], label="Margin Sampling")
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('30% labelling Stream Based')
plt.legend()
plt.show()

while len(x_stream_temp0) > 1000:

    stream_idx0 = np.random.choice(range(len(x_stream_temp0)), size=5)
    stream_idx1 = np.random.choice(range(len(x_stream_temp1)), size=5)
    stream_idx2 = np.random.choice(range(len(x_stream_temp2)), size=5)

    uncertainty_stream_0 = np.mean(
        [classifier_uncertainty(learnerstream0, x_stream_temp0[i].reshape(1, -1)) for i in stream_idx0])
    if uncertainty_stream_0 >= 0.3:
        X0, y0 = x_stream_temp0[stream_idx0].reshape(5, -1), y_stream_temp0[stream_idx0].reshape(5, )
        learnerstream0.teach(X=X0, y=y0)
    new_score0 = learnerstream0.score(x_unused, y_unused)
    uncertainty_stream_1 = np.mean(
        [classifier_uncertainty(learnerstream1, x_stream_temp1[i].reshape(1, -1)) for i in stream_idx1])
    if uncertainty_stream_1 >= 0.3:
        X1, y1 = x_stream_temp1[stream_idx1].reshape(5, -1), y_stream_temp1[stream_idx1].reshape(5, )
        learnerstream1.teach(X=X1, y=y1)
    new_score1 = learnerstream1.score(x_unused, y_unused)
    uncertainty_stream_2 = np.mean(
        [classifier_uncertainty(learnerstream2, x_stream_temp2[i].reshape(1, -1)) for i in stream_idx2])
    if uncertainty_stream_2 >= 0.3:
        X2, y2 = x_stream_temp2[stream_idx2].reshape(5, -1), y_stream_temp2[stream_idx2].reshape(5, )
        learnerstream2.teach(X=X2, y=y2)
    new_score2 = learnerstream2.score(x_unused, y_unused)

    performance_history_stream.append([new_score0, new_score1, new_score2])
    x_stream_temp0, y_stream_temp0 = np.delete(x_stream_temp0, stream_idx0, axis=0), np.delete(y_stream_temp0,
                                                                                               stream_idx0)
    x_stream_temp1, y_stream_temp1 = np.delete(x_stream_temp1, stream_idx1, axis=0), np.delete(y_stream_temp1,
                                                                                               stream_idx1)
    x_stream_temp2, y_stream_temp2 = np.delete(x_stream_temp2, stream_idx2, axis=0), np.delete(y_stream_temp2,
                                                                                               stream_idx2)
    print(performance_history_stream[-1])

performance_plot = [[], [], []]
for x in performance_history_stream:
    performance_plot[0].append(x[0])
    performance_plot[1].append(x[1])
    performance_plot[2].append(x[2])
# PLotting the 40% Stream Based Results
performance_plot = np.asarray(performance_plot)
plt.plot(performance_plot[0], label="Uncertainity Sampling")
plt.plot(performance_plot[1], label="Entropy Sampling")
plt.plot(performance_plot[2], label="Margin Sampling")
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('40% labelling Stream Based')
plt.legend()
plt.show()

import pickle
# Saving all the important files for later use at further stages
pkl_filename = "learnerpool0.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(learnerpool0, file)
pkl_filename = "learnerpool1.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(learnerpool1, file)
pkl_filename = "learnerpool2.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(learnerpool2, file)
pkl_filename = "learnerstream0.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(learnerstream0, file)
pkl_filename = "learnerstream1.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(learnerstream1, file)
pkl_filename = "learnerstream2.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(learnerpool2, file)
pkl_filename = "x_unused.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(x_unused, file)
pkl_filename = "y_unused.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(y_unused, file)

