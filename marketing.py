import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
sns.set_style('whitegrid', {'axes.grid' : False})

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

my_data = pd.read_csv("C:/Users/jhama/Documents/Python_Scripts/marketing/bank-additional-full.csv", delimiter=";")
print(my_data.head())

my_data.info()
my_data.dtypes

# Checking Null values in different columns
my_data.isnull().sum()
# There are no null values anywhere in the data

# Age
sns.distplot(my_data.age, kde = False, color='b', bins = 10)
plt.title('Distribution of Age')
plt.ylabel('Count')
plt.xlabel('Age(Years)')
plt.show()

# Job
g = sns.countplot(x= 'job', data = my_data)
plt.xticks(rotation = 45)
plt.title('Distribution of Jobs')
plt.show(g)

# Housing
sns.countplot(x = 'housing', data = my_data)
plt.title('Count of people wo have taken housing loan')
plt.show()

# Personal loan
sns.countplot(x = 'loan', data = my_data)
plt.title('Count of people wo have taken personal loan')
plt.show()

# Mode of Contact
sns.countplot(x = 'contact', data = my_data)
plt.title('Mode of contact')
plt.show()

# Duration
sns.distplot(my_data.duration/60, color='r')
plt.title('Call Duration')
plt.xlabel('Duration (minutes)')
plt.ylabel('Count')
plt.xlim(0,35)
plt.show()

# Whether subscribed
sns.countplot(x = 'y', data=my_data)
plt.title("Count of people who did and didn't subscribe")
plt.ylabel('Count')
plt.xlabel(' ')
plt.show()

# Let's Convert categorical data into numerical
le_job = preprocessing.LabelEncoder()
my_data['job'] = le_job.fit_transform(my_data['job'])

le_marital = preprocessing.LabelEncoder()
my_data['marital'] = le_marital.fit_transform(my_data['marital'])

le_education = preprocessing.LabelEncoder()
my_data['education'] = le_education.fit_transform(my_data['education'])

le_default = preprocessing.LabelEncoder()
my_data['default'] = le_default.fit_transform(my_data['default'])

le_housing = preprocessing.LabelEncoder()
my_data['housing'] = le_housing.fit_transform(my_data['housing'])

le_loan = preprocessing.LabelEncoder()
my_data['loan'] = le_loan.fit_transform(my_data['loan'])

le_contact = preprocessing.LabelEncoder()
my_data['contact'] = le_contact.fit_transform(my_data['contact'])

le_month = preprocessing.LabelEncoder()
my_data['month'] = le_month.fit_transform(my_data['month'])

le_day_of_week = preprocessing.LabelEncoder()
my_data['day_of_week'] = le_day_of_week.fit_transform(my_data['day_of_week'])

le_poutcome = preprocessing.LabelEncoder()
my_data['poutcome'] = le_poutcome.fit_transform(my_data['poutcome'])

le_y = preprocessing.LabelEncoder()
my_data['y'] = le_y.fit_transform(my_data['y'])
print(my_data.head())

#my_data.to_csv('mydata.csv')


# PCA
pca = PCA(n_components=2)
pca.fit(my_data)
x_PCA = pca.fit_transform(my_data)
print((x_PCA.shape))
pca_df = pd.DataFrame(x_PCA, columns=['PC1', 'PC2'])

#pca_df.to_csv('abc.csv')
print(pca_df.head())

print(pca.components_)
print(pca.explained_variance_ratio_)
#print(pca.singular_values)

X = my_data.iloc[:,0:19]
print(X.head())
Y = my_data.iloc[:,20]
print(Y.head())

# split data into train and test sets
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
clf = AdaBoostClassifier(n_estimators= 100, learning_rate= 1)

print("training...!")
clf.fit(X_train, y_train)

# make predictions for test data
print("predicting..!")
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#######################PCs##############################

a_train, a_test, b_train, b_test = train_test_split(pca_df, Y, test_size=test_size, random_state=seed)
clf.fit(a_train, b_train)

# make predictions for test data
y_pred = clf.predict(a_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(b_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

##############################################################################

pca_with = pd.concat([pca_df, my_data['y']], axis=1)
pca_with.columns = ['PC1', 'PC2', 'Successful']

sns.lmplot('PC1','PC2', hue = 'Successful', data=pca_with, fit_reg=False, scatter_kws={'alpha':0.5}, markers=["o", "x"], palette="Set1", x_jitter=.9)
plt.show()
