#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

print 'starting svm classification'

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

from sklearn import svm
clf = svm.SVC(kernel='rbf', C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)
print 'fitting time', time() - t0, 's'

# t1 = time()
# print clf.score(features_test, labels_test)
# print 'scoring time', time() - t1, 's'

# print clf.predict(features_test[10])
# print clf.predict(features_test[26])
# print clf.predict(features_test[50])

result = clf.predict(features_test)

chris_count = 0

for r in result:
    if r == 1:
        chris_count += 1

print chris_count
#########################################################


