#Basic and most important libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Classifiers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB

#Model evaluation tools
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

#Data processing functions

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

p = pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/hotel_bookings.csv")
print(p.head())

sns.countplot(x="hotel",hue="is_canceled",data=p)
plt.show()

sns.countplot(x="arrival_date_month",hue="is_canceled",data=p)
plt.show()

'''sns.countplot(x="country",hue="is_canceled",data=p)
plt.show()'''

sns.countplot(x="meal",hue="is_canceled",data=p)
plt.show()

sns.countplot(x="previous_cancellations",hue="is_canceled",data=p)
plt.show()

sns.countplot(x="is_repeated_guest",hue="is_canceled",data=p)
plt.show()
# Visualizing correlation through heatmap
correlation_mat = p.corr()
sns.heatmap(correlation_mat,annot=True,linewidths=.5,cmap="YlGnBu")
plt.show()

print(p.info())
print(p.isnull().sum())

#sns.heatmap(p.isnull(),yticklabels=False)
#plt.show()

p['children'] = p['children'].replace(np.nan,p['children'].median())
p['country'] = p['country'].replace(np.nan,p['country'].mode().values[0])
p['agent'] = p['agent'].replace(np.nan,p['agent'].median())
p['company'] = p['company'].replace(np.nan,p['company'].median())

print(p.isnull().sum())

sns.heatmap(p.isnull(),yticklabels=False)
plt.show()

print(p.duplicated().sum())

p.drop_duplicates(inplace = True)
print(p.duplicated().sum())

print(p.shape)

#Encoding

p['hotel'] = le.fit_transform(p['hotel'])
p['arrival_date_month'] = le.fit_transform(p['arrival_date_month'])
p['meal'] = le.fit_transform(p['meal'])
p['country'] = le.fit_transform(p['country'])
p['market_segment'] = le.fit_transform(p['market_segment'])
p['distribution_channel'] = le.fit_transform(p['distribution_channel'])
p['reserved_room_type'] = le.fit_transform(p['reserved_room_type'])
p['assigned_room_type'] = le.fit_transform(p['assigned_room_type'])
p['deposit_type'] = le.fit_transform(p['deposit_type'])
p['customer_type'] = le.fit_transform(p['customer_type'])
p['reservation_status'] = le.fit_transform(p['reservation_status'])
p['reservation_status_date'] = le.fit_transform(p['reservation_status_date'])

print(p.info())

p['y'] = p['is_canceled']
p = p.drop(['is_canceled'],axis = 1)
print(p.info())

print(p)

X = p.iloc[:, :-1].values
y = p.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Logistic Regression
model=LogisticRegression(solver="liblinear")
model.fit(X_train,y_train)

print('Training Data Accuracy Score',model.score(X_train,y_train))
print('Testing Data Accuracy Score',model.score(X_test,y_test))

#Decision Tree
dtree=DecisionTreeClassifier(criterion = 'gini', max_depth = 1, random_state=0)
dtree.fit(X_train,y_train)

print('Training Data Accuracy Score',dtree.score(X_train,y_train))
print('Testing Data Accuracy Score',dtree.score(X_test,y_test))


#KNN 
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

print('Training Data Accuracy Score',knn.score(X_train,y_train))
print('Testing Data Accuracy Score',knn.score(X_test,y_test))

#Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

print('Training Data Accuracy Score',nb.score(X_train,y_train))
print('Testing Data Accuracy Score',nb.score(X_test,y_test))

#AdaBoost CLassifier
abcl = AdaBoostClassifier(n_estimators = 120,random_state=0)
abcl = abcl.fit(X_train, y_train)

print('Training Data Accuracy Score',abcl.score(X_train,y_train))
print('Testing Data Accuracy Score',abcl.score(X_test,y_test))

#Logistic regression confusion matrix
y1 = model.predict(X_test)
cm1 = confusion_matrix(y_test, y1,labels=[0, 1])
df_cm1 = pd.DataFrame(cm1, index = [i for i in ["No","Yes"]],columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm1, annot=True ,fmt='g')
plt.show()

print("Classification Report")
print(classification_report(y_test, y1, labels=[1, 0]))

'''#Decision Tree confusion matrix
y2 = dtree.predict(X_test)
cm2 = confusion_matrix(y_test, y2,labels=[0, 1])
df_cm2 = pd.DataFrame(cm2, index = [i for i in ["No","Yes"]],columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm2, annot=True ,fmt='g')
plt.show()

print("Classification Report")
print(classification_report(y_test, y2, labels=[1, 0]))'''

#KNN confusion matrix
y3 = knn.predict(X_test)
cm3 = confusion_matrix(y_test, y3,labels=[0, 1])
df_cm3 = pd.DataFrame(cm3, index = [i for i in ["No","Yes"]],columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm3, annot=True ,fmt='g')
plt.show()

print("Classification Report")
print(classification_report(y_test, y3, labels=[1, 0]))

#Naive Bayes confusion matrix
y4 = nb.predict(X_test)
cm4 = confusion_matrix(y_test, y4,labels=[0, 1])
df_cm4 = pd.DataFrame(cm4, index = [i for i in ["No","Yes"]],columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm4, annot=True ,fmt='g')
plt.show()

print("Classification Report")
print(classification_report(y_test, y4, labels=[1, 0]))

#AdaBoost confusion matrix
y5 = abcl.predict(X_test)
cm5 = confusion_matrix(y_test, y5,labels=[0, 1])
df_cm5 = pd.DataFrame(cm5, index = [i for i in ["No","Yes"]],columns = [i for i in ["No","Yes"]])
sns.heatmap(df_cm5, annot=True ,fmt='g')
plt.show()

print("Classification Report")
print(classification_report(y_test, y5, labels=[1, 0]))
