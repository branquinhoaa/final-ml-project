#!/usr/bin/python


"""
    In this code
"""

import pickle
import sys
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

################################
# here I split the data into test and training data

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

################################
# here I create my decision tree classifier

from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features_train, labels_train)

accuracy = clf.score(features_test, labels_test)

# here I show the accuracy of my data:

print("accuracy: ", accuracy)
print(sum(labels_test))
print(sum(labels_train))
print(len(labels_test))
print(len(labels_train))


