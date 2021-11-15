""" Script for extracting decision trees rules into SAS code"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tree_to_sas import get_rules


# Example with iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Get SAS rules for all trees
sas_tree_rules = {}
for i, tree in enumerate(rf):
    tree_name = "TREE_{}".format(i)
    tree_rules = get_rules(tree=rf[i], tree_id=i, features=iris.feature_names, sas_table="DATASET")
    sas_tree_rules[tree_name] = tree_rules

# Output of first Tree rules in SAS
print(sas_tree_rules["TREE_0"])

