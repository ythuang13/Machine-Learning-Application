#-------------------------------------------------------------------------
# AUTHOR: Yitian Huang
# FILENAME: question4.py
# SPECIFICATION: Repurpose question 5 code for question 4
# FOR: CS 4210- Assignment #5
# TIME SPENT: 20 mins
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#read the dataset using pandas
df = pd.read_csv('question4_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

# encode item for each transaction
encoded_vals = []
for index, row in df.iterrows():
    labels = {}
    for item in itemset:
        labels[item] = 0

    for item in row:
        if item is not np.nan:
            labels[item] = 1

    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.3, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

ranking = []
for index, row in freq_items.iterrows():
    ranking.append((set(row.get("itemsets")),row.get("support")))
ranking = sorted(ranking, key=lambda x: -x[1])
for itemset in ranking:
    print(f"s({itemset[0]})={itemset[1]}")

ranking = []
for index, row in rules.iterrows():
    ranking.append((set(row.get("antecedents")),set(row.get("consequents")),row.get("confidence")))

ranking = sorted(ranking, key=lambda x: -x[2])
for itemset in ranking:
    if len(itemset[0]) + len(itemset[1]) == 3:
        print(f"c({itemset[0]}->{itemset[1]})={itemset[2]}")

#iterate the rules data frame and print the apriori algorithm results by using the following format:
# for index, rule in rules.iterrows():
#     antecedents_list = list(rule.get('antecedents'))
#     print(antecedents_list[0], end="")
#     for item in antecedents_list[1:]:
#         print(f", {item}", end="")
#     print(" -> ", end="")
#     consequents_list = list(rule.get("consequents"))
#     print(consequents_list[0], end="")
#     for item in consequents_list[1:]:
#         print(f", {item}", end="")
#     print()
    
#     print(f"Support: {rule.get('support')}")
#     print(f"Confidence: {rule.get('confidence')}")

#     #To calculate the prior and gain in confidence, find in how many transactions
#     # the consequent of the rule appears (the supporCount below). Then,
#     #use the gain formula provided right after.
#     #prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#     #print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#     #-->add your python code below
#     rule_confidence = rule.get('confidence')
#     prior = rule.get('consequent support')
#     print(f"Prior: {prior}")
#     print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#     print()

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
# plt.show()