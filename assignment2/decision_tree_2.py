#-------------------------------------------------------------------------
# AUTHOR: Yitian Huang
# FILENAME: decision_tree_2
# SPECIFICATION: Use different dataset to generate different decision trees
# and use testing date set to compare their accuracy.
# FOR: CS 4210- Assignment #2
# TIME SPENT: Roughly 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

def lens_label_encoder(subject) -> list:
    result = []

    age = subject[0]
    if age == "Young":
        result.append(1)
    elif age == "Prepresbyopic":
        result.append(2)
    elif age == "Presbyopic":
        result.append(3)
    
    # Spectacle Prescription, Myope = 1, Hypermetrope = 2
    spectacle = subject[1]
    if spectacle == "Myope":
        result.append(1)
    elif spectacle == "Hypermetrope":
        result.append(2)

    # Astigmatism, No = 1, Yes = 2
    astigmatism = subject[2]
    if astigmatism == "No":
        result.append(1)
    elif astigmatism == "Yes":
        result.append(2)
    
    # TPR, Normal = 1, Reduced = 2
    tpr = subject[3]
    if tpr == "Normal":
        result.append(1)
    elif tpr == "Reduced":
        result.append(2)

    return result

ds_counter = 0
for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    for row in dbTraining:
        result = lens_label_encoder(row)
        X.append(result)

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> addd your Python code here
    for row in dbTraining:
        # label, Yes = 1, No = 2
        if row[-1] == "Yes":
            Y.append(1)
        else:
            Y.append(2)

    #loop your training and test tasks 10 times here
    min_accuracy = 1
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = []
        with open("contact_lens_test.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append(row)

        true_pos = true_neg = false_pos = false_neg = 0
        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            test_features = lens_label_encoder(data)
            class_predicted = clf.predict([test_features])[0]

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            
            test_label = data[4]
            if test_label == "Yes":
                if class_predicted == 1:
                    false_neg += 1
                else:
                    true_pos += 1
            else:
                if class_predicted == 1:
                    true_neg += 1
                else:
                    false_pos += 1
        
        # print(f"{true_pos=} {true_neg=} {false_pos=} {false_neg=}")
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        # print(f"{accuracy=}")

        #find the lowest accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
        if accuracy < min_accuracy:
            min_accuracy = accuracy

    #print the lowest accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"final accuracy when training on {dataSets[ds_counter]}: {min_accuracy}")
    ds_counter += 1


