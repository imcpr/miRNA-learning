import numpy as np
import csv
import os
import cPickle
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [0, 1], rotation=45)
    plt.yticks(tick_marks, [0, 1])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

data = cPickle.load(open('all_data.pkl', 'rb'))


X = data[:, :-1]
Y = np.ravel(data[:, -1:])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# add synthetic SMOTE examples
# X_train = np.vstack((X_train, synth[:, :-1]))
# Y_train = np.concatenate((Y_train, np.ravel(synth[:, -1:])))
# for row in X_train:
#     if row[0] > 0:
#         row[0] = 1

sampledX = cPickle.load(open('5050_sampled_data.pkl', 'rb'))

X_train = sampledX[:, :-1]
Y_train = sampledX[:, -1:].flatten()

save_pics = True


#SSTP
sstp_preds = X_test[:, :1]
sstp_preds = sstp_preds.clip(min=0, max=1)

print "Using SSTP ---"
print "Zero one error rate % f" % zero_one_loss(Y_test, sstp_preds)
print "Accuracy score % f" % accuracy_score(Y_test, sstp_preds)
print "Confusion matrix "
print confusion_matrix(Y_test, sstp_preds)
print(classification_report(Y_test, sstp_preds))
if save_pics:
    plt.figure()
    cm = confusion_matrix(Y_test, sstp_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.savefig('figures/sstp_confusion_matrix.png', format='png')



# naive_bayes

print "Fitting Naive Bayes Classifer..."
nb = BernoulliNB()
nb = nb.fit(X_train, Y_train)
nb_preds = nb.predict(X_test)

print "Using Naive Bayes ---"
print "Zero one error rate % f" % zero_one_loss(Y_test, nb_preds)
print "Accuracy score % f" % accuracy_score(Y_test, nb_preds)
print "Confusion matrix "
print confusion_matrix(Y_test, nb_preds)
print(classification_report(Y_test, nb_preds))

if save_pics:
    plt.figure()
    cm = confusion_matrix(Y_test, nb_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.savefig('figures/nb_confusion_matrix.png', format='png')


# KNN
# print "Fitting KNN Classifer..."
# knn = KNeighborsClassifier()
# knn = knn.fit(X_train, Y_train)
# knn_preds = nb.predict(X_test)

# print "Using KNN ---"
# print "Zero one error rate % f" % zero_one_loss(Y_test, knn_preds)
# print "Accuracy score % f" % accuracy_score(Y_test, knn_preds)
# print "Confusion matrix "
# print confusion_matrix(Y_test, knn_preds)
# print(classification_report(Y_test, knn_preds))

# if save_pics:
#     plt.figure()
#     cm = confusion_matrix(Y_test, knn_preds)
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
#     plt.savefig('figures/knn_confusion_matrix.png', format='png')




#LogisticRegression

print "Fitting LogisticRegression Classifer..."
lgr = LogisticRegression()
lgr = lgr.fit(X_train, Y_train)
lgr_preds = lgr.predict(X_test)

print "Using LogisticRegression ---"
print "Zero one error rate % f" % zero_one_loss(Y_test, lgr_preds)
print "Accuracy score % f" % accuracy_score(Y_test, lgr_preds)
print "Confusion matrix "
print confusion_matrix(Y_test, lgr_preds)
print(classification_report(Y_test, lgr_preds))

if save_pics:
    plt.figure()
    cm = confusion_matrix(Y_test, lgr_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.savefig('figures/lgr_confusion_matrix.png', format='png')

print "Fitting RF Classifer..."
ada = RandomForestClassifier(n_estimators=20)
ada = ada.fit(X_train, Y_train)
ada_preds = ada.predict(X_test)

print "Using RF ---"
print "Zero one error rate % f" % zero_one_loss(Y_test, ada_preds)
print "Accuracy score % f" % accuracy_score(Y_test, ada_preds)
print "Confusion matrix "
print confusion_matrix(Y_test, ada_preds)
print(classification_report(Y_test,ada_preds))

if save_pics:
    plt.figure()
    cm = confusion_matrix(Y_test, ada_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.savefig('figures/rf_confusion_matrix.png', format='png')

print "Fitting ERF Classifer..."
erf = ExtraTreesClassifier(n_estimators=30)
erf = erf.fit(X_train, Y_train)
erf_preds = erf.predict(X_test)

print "Using ERF ---"
print "Zero one error rate % f" % zero_one_loss(Y_test, erf_preds)
print "Accuracy score % f" % accuracy_score(Y_test, erf_preds)
print "Confusion matrix "
print confusion_matrix(Y_test, erf_preds)
print(classification_report(Y_test,erf_preds))

if save_pics:
    plt.figure()
    cm = confusion_matrix(Y_test, erf_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.savefig('figures/erf_confusion_matrix.png', format='png')


# print "Fitting Ada Classifer..."
# ada = AdaBoostClassifier(n_estimators=100)
# ada = ada.fit(X_train, Y_train)
# ada_preds = ada.predict(X_test)

# print "Using ADA ---"
# print "Zero one error rate % f" % zero_one_loss(Y_test, ada_preds)
# print "Accuracy score % f" % accuracy_score(Y_test, ada_preds)
# print "Confusion matrix "
# print confusion_matrix(Y_test, ada_preds)
# print(classification_report(Y_test,ada_preds))

# if save_pics:
#     plt.figure()
#     cm = confusion_matrix(Y_test, ada_preds)
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
#     plt.savefig('figures/ada_confusion_matrix.png', format='png')




# print "Fitting SVM Classifer..."
# svm = SVC()
# svm = svm.fit(X_train, Y_train)
# svm_preds = svm.predict(X_test)

# print "Using SVM ---"
# print "Zero one error rate % f" % zero_one_loss(Y_test, svm_preds)
# print "Accuracy score % f" % accuracy_score(Y_test, svm_preds)
# print "Confusion matrix "
# print confusion_matrix(Y_test, svm_preds)
# if save_pics:
#     plt.figure()
#     cm = confusion_matrix(Y_test, svm_preds)
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
#     plt.savefig('figures/svm_confusion_matrix.png', format='png')