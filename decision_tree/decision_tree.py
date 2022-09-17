#-------------------------------------------------------------------------
# AUTHOR: Yitian Huang
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #1
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
for row in db:
	result = []

	# Age, Young = 1, Prepresbyopic = 2, Presbyopic = 3
	age = row[0]
	if age == "Young":
		result.append(1)
	elif age == "Prepresbyopic":
		result.append(2)
	elif age == "Presbyopic":
		result.append(3)
	
	# Spectacle Prescription, Myope = 1, Hypermetrope = 2
	spectacle = row[1]
	if spectacle == "Myope":
		result.append(1)
	elif spectacle == "Hypermetrope":
		result.append(2)

	# Astigmatism, No = 1, Yes = 2
	astigmatism = row[2]
	if astigmatism == "No":
		result.append(1)
	elif astigmatism == "Yes":
		result.append(2)
	
	# TPR, Normal = 1, Reduced = 2
	tpr = row[3]
	if tpr == "Normal":
		result.append(1)
	elif tpr == "Reduced":
		result.append(2)

	X.append(result)

#transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
for row in db:
	# label, Yes = 1, No = 2
	if row[-1] == "Yes":
		Y.append(1)
	else:
		Y.append(2)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy', )
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()