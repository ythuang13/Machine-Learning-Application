#-------------------------------------------------------------------------
# AUTHOR: Yitian Huang
# FILENAME: naive_bayes.py
# SPECIFICATION: Implement naive bayes
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

db = []

#reading the training data in a csv file
#--> add your Python code here
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook = {
    "Sunny": 0,
    "Overcast": 1,
    "Rain": 2
}
temperature = {
    "Cool": 0,
    "Mild": 1,
    "Hot": 2
}
humidity = {
    "Normal": 0,
    "High": 1
}
wind = {
    "Weak": 0,
    "Strong": 1
}

X = []
for sample in db:
    X.append([outlook[sample[1]], temperature[sample[2]],
        humidity[sample[3]], wind[sample[4]]])

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for sample in db:
    Y.append(1 if sample[5] == "Yes" else 2)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
test_db = []
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         test_db.append(row)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for sample in test_db:
    confidences = clf.predict_proba([[
        outlook[sample[1]],
        temperature[sample[2]],
        humidity[sample[3]],
        wind[sample[4]]]])[0]

    prediction = "NA"
    confidence = None
    if confidences[0] >= 0.75:
        prediction = "Yes"
        confidence = confidences[0]
    elif confidences [1] >= 0.75:
        prediction = "No"
        confidence = confidences[1]
        
    print (sample[0].ljust(15) + sample[1].ljust(15) + sample[2].ljust(15)
        + sample[3].ljust(15) + sample[4].ljust(15) + prediction.ljust(15)
        + str(confidence).ljust(15))

