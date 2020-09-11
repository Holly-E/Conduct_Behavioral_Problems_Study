"""
Erickson, Holly
Conduct and Behavioral Problems in Children Study
"""

# Read SAS file using pandas

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

"""
Keep 2
RF, LG, SVM: remove conduct feats - replace 6, 7 info - conduct = 1,0
"""

#%%
# Read the data
filename = "x_keep_4_2016_males"
x = pd.read_csv(filename + ".csv")
Y = pd.read_csv('Y4_2016.csv')

#x_rem_conduct = x.drop('K2Q34A', axis = 1)
print(Y.K2Q34A.value_counts())  
#print(x.dtypes)

x = x.select_dtypes(include=['float64'])

# %%
# """
# Normalize data 
# """
# from sklearn import preprocessing
# normalized_x = preprocessing.normalize(x)

#%%
np.random.seed(123)
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size = 0.2)
#%%
"""
Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=200, 
                               bootstrap = True,
                               max_features = 'sqrt',
                               n_jobs = -1)

# get predictions for each row using cross validation 
y_pred = cross_val_predict(model, x, Y.values.ravel(), cv=3)
x['predictions'] = y_pred
x['target'] = Y['K2Q34A']


#%%
"""
SVC
"""
model = SVC() # The default kernel used by SVC is the gaussian kernel

#%%
"""
To avoid warning:  A column-vector y was passed when a 1d array was expected.
.values will give the values in an array. (shape: (n,1)
.ravel will convert that array shape to (n, )
"""
# Fit on training data
model.fit(x_train, y_train.values.ravel())


prediction = model.predict(x_test)
cm = confusion_matrix(y_test, prediction)
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]
    
accuracy = sum/x_test.shape[0]
print(accuracy)

"""
2012
Keep 2
RF - 0.966
SVM - 0.960

Keep 3
RF - 0.978
SVM - 0.974\
    

Keep 4
RF - 0.959
SVM - 0.949

---
2016
Keep 2
RF - 0.951
SVM - 0.917

Keep 3 
RF - 0.966
SVM - 0.952

Keep 4
RF - 0.934
SVM - 0.887

---
2017
Keep 2
RF - 0.942
SVM - 0.91

Keep 3 
RF - 0.96
SVM - 0.948

Keep 4
RF - 0.931
SVM - 0.885

---
2018
Keep 2
RF - 0.947
SVM - 0.920

Keep 3 
RF - 0.968
SVM - 0.960

Keep 4
RF - 0.926
SVM - 0.887
"""

#%%
"""
Random Forest
Get prediction probabilities and ROC AUC 
"""
from sklearn.metrics import roc_auc_score

# Probabilities for each class
rf_probs = model.predict_proba(x_test)[:, 1]

# Calculate roc auc
roc_value = roc_auc_score(y_test, rf_probs)
print("ROC Value")
print(roc_value)

#%%
# Extract feature importances
fi = pd.DataFrame({'feature': list(x_test.columns),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)

# Display and Save
print(fi.head())
fi.to_csv("keep_4_2016_RF_Feat_imp.csv", index = False)
x.to_csv("keep_4_2016_RF_pred.csv", index = False)

#%%
"""
SVM
Get precision - recall score 
"""
from sklearn.metrics import average_precision_score
y_score = model.decision_function(x_test)
average_precision = average_precision_score(y_test, y_score)
print("Average Precision")
print(average_precision)
print("Confusion Matrix")
print(cm)


#%%

# list_in_order = ['prediction','y_test', 'y_score']
# for ind in range(len(list(x_test)) - 3):
#     list_in_order.append(list(x_test)[ind])
  
# output = pd.DataFrame(x_test, columns = list_in_order)

#%%
#df1_scores = x_test.drop(['prediction_df', 'y_score_df'], axis = 1)

#Modify keep codes if used (1, 2, 3 currently have been labeled)
# output.to_csv(filename + ".csv", index = False)


#%%
"""
#2. Fit and Evaluate the Model
#Fit the model as a logistic regression model with the following parameters.
# LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8).
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(df_final)

# Provide the area under the ROC curve for the model.
trainingSummary = lrModel.summary
print('Area Under ROC: ' + str(trainingSummary.areaUnderROC))

Output
Area Under ROC: 0.5
"""
#%%
"""
Unsupervised Learning
"""
not_norm = pd.read_csv('keep_15_no_conduct_only_ADHD_code_2.csv')
x = pd.read_csv('keep_15_NORM_no_conduct_only_ADHD_code_2.csv')

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k, n_jobs = -1)
    km = km.fit(x)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#%%
kmeans = KMeans(n_clusters=3, random_state=0, n_jobs=-1).fit(x)
labels = kmeans.labels_
#label_df = pd.DataFrame(labels, columns = ['labels'])

x_cols = list(x)
not_norm['label'] = labels
print(not_norm['label'].value_counts())

centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(data = centroids, columns = x_cols)

centroid_df.to_csv('centroids_3_keep_15.csv', index = False)

#%%
"""
Plot K2Q34A and K6Q06 colored by label
"""
import seaborn
import matplotlib.pyplot as plt
seaborn.set(style='ticks')

fg = seaborn.FacetGrid(data=not_norm, hue='label', col = 'SEX', row= 'K2Q34A', aspect=1.61)
fg.map(plt.plot, 'K6Q06', 'K2Q34A').add_legend()

#g = sns.FacetGrid(tips, col="time",  row="smoker")
#g = g.map(plt.scatter, "total_bill", "tip", edgecolor="w")

Oooooooooooooooooooooooo\'/#%%
not_norm.to_csv('keep_11_2_clus_w_conduct_only_ADHD_codes_12.csv', index = False)