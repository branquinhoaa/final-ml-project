#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### The first step is select the features I will use. 
# All features available to use are: 
# financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 
# email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
# POI_label = ['poi']

# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

financial_features = ['salary', 'bonus', 'director_fees'] 
email_features = ['from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
POI_label = ['poi']
features_list = []
features_list += POI_label + financial_features  

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### The second step on my poi identifier is remove the data outliers, or the points that really is out of the 
# ordinary values. These kind of data can change my results and I want to know how is the data behaviour in general.

# TODO: VERIFY THE BEST WAY TO IDENTIFY THE OUTLIERS


### The third step is to create a new feature based in what do you think that makes sense.
# for me, makes sense to create a feature that sums up all the emails exchanged between a person and a poi.
# so my new feature means something like how close is the communication between a person and a poi and 
# my hypotesis is that highest this feature is, highest a chance of a person be a poi too.
communication_feature_list = ['poi', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']

comm_data = featureFormat(data_dict, communication_feature_list)

comm_labels, comm_features = targetFeatureSplit(comm_data)


################## CORRELATION CHECK #############################
# check for correlations between any pair of communication feature
# plot a graph to analyse the correlation
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib.pyplot as plt

# palette = {0 : 'blue', 1 : 'red'}
# comm_correlation = map(lambda x: palette[int(x)], comm_labels)

new_data_frame = pd.DataFrame(comm_features, columns=communication_feature_list[1:])
collor_map = cm.get_cmap('Spectral')
grr = pd.plotting.scatter_matrix(new_data_frame, alpha=0.8, c=comm_labels, cmap=collor_map)
#plt.show()
####################################################################
# NOW I WILL INCLUDE MY NEW FEATURE ON THE DATA SET
for key, value in data_dict.items():
	print(key, value)
	break
print(data_dict[0])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)