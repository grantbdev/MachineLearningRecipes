from sklearn import tree

# This program shows a basic example of machine learning
# The goal is to distinguish between apples and oranges

# The features are weight in grams (integer value) and texture (0 for bumpy, 1 for smooth)
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# Each item in this array corresponds to a list of features in the features array
# 0 for apple, 1 for orange
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# Predict whether something that is bumpy and 160 grams is an apple or an orange
print clf.predict([[160, 0]])