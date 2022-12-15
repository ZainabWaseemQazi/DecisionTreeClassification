# Pythonpro
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

print("Grip: The Spark Foundation")
print("Zainab Waseem Qazi")
print("Data Science and Business Analytics Interee")

#Loading data
data_link="http://bit.ly/w-data"
stu_data=pd.read_csv(data_link)
print("Successfully inported data at this step")

#displaying first the data
print(stu_data.head())
print(stu_data.tail())

#Displaying data through graph with dependant and independant variables
stu_data.plot(x='Hours', y='Scores', style='*')
plt.title('Graph of Studied hours effect on percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scored')
plt.show()

#Divide the data in Inputs and Outputs
print("Identifying inputs and outputs from data")
X=stu_data.iloc[:, :-1].values #including Hours as input
y=stu_data.iloc[:,1].values #including Scores as output


#Spliting data into testing and training sets
print("Splitting data into test and train sets")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=0)


#Training Algorithm
print("Training our Linear Regression Algorithm")
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train, y_train)
print("Model is trained")

plt_line=reg.coef_*X+reg.intercept_
plt.scatter(X,y)
plt.plot(X,plt_line);
plt.show()

#Prediction
test_data=np.array(X_test)
print("Predicting data")
predict_val= reg.predict(test_data)
# Predicting the scores
df=pd.DataFrame({'Actual Score': y_test, 'Predicted Score': predict_val})
print(df)

print("Now predicting our own data let's suppose hours is 9.25 then predict the scores")
hours=9.25
hours=np.array(hours)
hours=hours.reshape(1,-1)
test_pred=reg.predict(hours)
print("Hours ", hours)
print("Predicted Score", test_pred)

from sklearn import metrics
print('Mean Absolute Error:',
      metrics.mean_absolute_error(y_test, predict_val))
exit()

