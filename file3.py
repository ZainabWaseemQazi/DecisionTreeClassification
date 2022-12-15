import pandas as pd
import sklearn.datasets as data_set
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import metrics

print("Grip: The Spark Foundation")
print("Zainab Waseem Qazi")
print("Data Science and Business Analytics Interee")

#Importing dataset
iris=data_set.load_iris()
from sklearn.model_selection import train_test_split
df=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=1)
print(df.head(5))
print(df.tail(5))
print("Data Loaded Successfully")
print("Below is our targeted attribute")
print(y)

#Creating Decision tree classifer
from sklearn.tree import DecisionTreeClassifier
dec_tree=DecisionTreeClassifier()
dec_tree.fit(X_train,y_train)
print("Decision Tree Classifier created")
fig = plt.figure(figsize=(10,6))
_ = tree.plot_tree(dec_tree,
                   feature_names=iris.feature_names,
                   class_names=iris.target_names,
                   filled=True)
plt.show()
fig.savefig("decistion_tree.png")

#predicting results
print("Predicting our remaining data")
y_output=dec_tree.predict(X_test)
df=pd.DataFrame({'Actual results': y_test, 'Predicted results': y_output})
print(df)

#Accuracy calculation
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_output))


exit()
