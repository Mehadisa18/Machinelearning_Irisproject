
# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#we use panda for descriptive statistics and data visualization
# Here we give column names

url="iris_data.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url,names=names)
#print(dataset)

# Lets see how certain functions from these libraries are useful
print(dataset.shape) # Gives the dimensions of data set

print(dataset.groupby('class').size()) # gives out put accordingly

print(dataset.describe())

print(dataset.head(20)) # shows top 20

# We normally use box and whisker plots to find out the data which is frequent. for example if we consider petal size, a box and whisker plot
# gives u the idea of the range in which the sizes of the petals for diff flowers lie.

# Unvariate -- for individual variables
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Histogram
dataset.hist()
plt.show()

# Multivariate plots
# scatter matrix
scatter_matrix(dataset)
plt.show()

# Dealing with algorithms



# Split-out validation dataset
array = dataset.values
print(array)
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# seed is a initial push.
# seed element determines and regulates the random numbers that are being produced in the future
# we have taken 20% of the values for validation whether the computer is predicting correctly or not

# Now let us explore a set of linear and non linear models

# Spot Check Algorithms
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Now we see pictorially as to which model is more accurate

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Now that we are almost sure that KNN suits better we validate it with validation data
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
# The accuracy score is 90% or 0.9
# The details about all these confusion matrix and classification report has to be studied