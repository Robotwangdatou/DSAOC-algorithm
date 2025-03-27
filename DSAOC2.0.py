# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:21:20 2024
DSAOC Algorithm Implementation with five-fold cross-validation repeated 10 times to calculate averages
@author: robot
"""
import time
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

#Functions for classification and accuracy metrics calculation*********************************************************************************************************************
 # Calculate G-mean
def g_mean_score(y_true, y_pred):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Calculate sensitivities for each class (recall)
    sensitivities = np.diag(cm) / np.sum(cm, axis=1)
    #  Calculate G-mean
    g_mean = np.sqrt(np.prod(sensitivities))
    return g_mean
 #Calculate accuracy metrics
def calculate_metrics(y, y_predict,Jresult):
     #print("the result of sklearn package")
     auc = roc_auc_score(y, y_predict)
     #print("sklearn auc:",auc)
     g_mean= g_mean_score(y, y_predict)
     #print("sklearn accuracy:",accuracy)
     recal = recall_score(y, y_predict)
     precision = precision_score(y, y_predict)
     F1_sc=(2*recal*precision)/(recal+precision)
     new_r=[auc,g_mean,F1_sc]
     Jresult.extend(new_r)
     
 #Build classification function and accuracy metrics calculation
def fenlei(x_train,y_train,x_test,y_test):

   #Create a list to store various accuracy metrics
   Jresult=[]
    
   #%%GaussianNB Classifier
   from sklearn.naive_bayes import GaussianNB
   gnb = GaussianNB()   # Initialize Naive Bayes with default settings
   gnb.fit(x_train,y_train)    # Fit the model parameters using training data
   calculate_metrics(y_test,gnb.predict(x_test),Jresult) 
   #%%Logistic Regression Classifier
   from sklearn.linear_model import LogisticRegression as LR
   lr = LR(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
           intercept_scaling=1, class_weight=None, random_state=None,
           solver='liblinear', max_iter=100, multi_class='ovr',
           verbose=0, warm_start=False, n_jobs=1) #Build logistic regression model
   lr.fit(x_train, y_train) 
   calculate_metrics(y_test,lr.predict(x_test),Jresult) 
    
   #%% Decision Tree Classifier
   from sklearn.tree import DecisionTreeClassifier as DTC
   dtc = DTC(criterion="gini",splitter="best",max_depth=None,min_samples_split=2,
                    min_samples_leaf=1,min_weight_fraction_leaf=0.,max_features=None,
                    random_state=None,max_leaf_nodes=None,class_weight=None) # Build decision tree model
   dtc.fit(x_train, y_train) 
   calculate_metrics(y_test,dtc.predict(x_test),Jresult)  
   
   return Jresult
#Spatial Computing Subfunctions*******************************************************************************************************************************************************************
#Function: Calculate the distance between two points
def getDistanceByEuclid(instance1, instance2):
    dist = 0
    for key in range(len(instance1)):
        dist += (float(instance1[key]) - float(instance2[key])) ** 2
    return dist ** 0.5
#Function: Calculate the local density difference of a single minority class point
def cutoff_kernel(distanceMartix,distanceMartixA,i,dc1):#Calculate density difference
    tempDensityA = 0#Number of majority class instances within radius
    tempDensityI=0# Number of minority class instances within radius
    tempDensity = 0#Local density difference within radius
 #Calculate the number of minority class instances within radius
    for j in range(len(distanceMartix[i])):
        tempDistance = distanceMartix[i][j]
        if tempDistance < dc1:
           tempDensityI += 1      
 # Calculate the number of majority class instances within radius
    for j in range(len(distanceMartixA[i])):
        tempDistanceA = distanceMartixA[i][j]
        if tempDistanceA < dc1:
           tempDensityA += 1   
    # Calculate density difference
    tempDensity =tempDensityI-tempDensityA
    return  tempDensity,tempDensityI,tempDensityA
#Function: Calculate the local density difference of a set of minority class points
def computeDensities(distanceMartix,distanceMartixA,dc1):
    temp_local_density_list = []
    temp_local_densityI_list=[] 
    temp_local_densityA_list=[]
    temp_local_density_listX=[] 
    for i in range(0, len(distanceMartix)):
        result1,result2,result3=cutoff_kernel(distanceMartix,distanceMartixA,i,dc1)
        temp_local_density_list.append(result1)
        temp_local_densityI_list.append(result2)
        temp_local_densityA_list.append(result3)
        temp_local_density_listX.append(round(result2/(result2+result3),2))
    return temp_local_density_list,temp_local_densityI_list,temp_local_density_listX,temp_local_densityA_list


#Function: Search for points closest to the target point within a given radius and return the indices of these points. Implementation does not rely on the kneighbors method. The function will use Euclidean distance as the distance measure.
def search_points_within_radius(data, target_point, radius):
    distances = np.linalg.norm(data - target_point, axis=1)  
    within_radius_indices = np.where(distances <= radius)[0] 
    return within_radius_indices

# New Sample Synthesis Algorithm Function Based on Overlapping Clusters*********************************************************************************************************************
class N_Smote(object):
      def __init__(self, N=50, r=0.5):
          # Initialize self.N, self.newindex
          self.N = N  # Sampling number
          self.r = r
          # self.newindex is used to record the number of samples already synthesized by the SMOTE algorithm
          self.newindex = 0

      # Build the training function
      def fit(self, Tsamples, mdc):  # Tsamples: minority class set, mdc: density difference matrix
          # Initialize self.samples, self.T, self.numattrs
          # self.T is the number of minority class samples, self.numattrs is the number of features of minority class samples
          self.T, self.numattrs = Tsamples.shape
          # Assume we have an index array
          indices = np.arange(len(Tsamples))
          # Assign weights to each point based on density difference for sampling
          global probability, nnarray
          probability = []
          probability1 = []
          for i in range(len(Tsamples)):
              if Tsamples[i][-2] >= 0.5 and Tsamples[i][-1] == 2:
                  probability.append(1 / Tsamples[i][-3])
                  probability1.append(1 / Tsamples[i][-2])
              else:
                  probability.append(1 / Tsamples[i][-3])
                  probability1.append(0.00000001)
          self.S_indices = random.choices(indices, probability1, k=self.N)  # Root sample index

          # Create an array to save the synthesized samples
          self.synthetic = np.zeros((self.N, self.numattrs))
          TTsamples = Tsamples[:, :-8]

          # Loop through all input samples
          for i in range(len(self.S_indices)):
              # Search all minority class points in the neighborhood and store them in an array
              nnarray = search_points_within_radius(TTsamples, TTsamples[self.S_indices[i]].reshape(1, -1), mdc)

              # Input N, i, nnarray into the sample synthesis function self.__populate
              self.__populate(i, nnarray, Tsamples, probability)

          # Finally return the synthesized samples self.synthetic
          return self.synthetic

      # Build the sample synthesis function
      def __populate(self, i, nnarray, TTsamples, probability):

          # Assign weights to each point based on density difference for sampling
          global cy
          cy = []
          for j in range(len(nnarray)):
              cy.append(probability[nnarray[j]])  # Select the relative density difference of each subpoint
          
          global nn
          if len(nnarray) == 1:
              nn = random.choices(nnarray, cy, k=1)
              # Calculate the difference
              diff = TTsamples[nn] - TTsamples[self.S_indices[i]]
              # Generate a random number between 0 and 1
              gap = random.uniform(0, 1)
              # Put the synthesized new sample into the array self.synthetic
              self.synthetic[self.newindex] = TTsamples[self.S_indices[i]] + gap * diff

          else:
              nn = random.choices(nnarray, cy, k=2)
              # Calculate the difference
              diff = TTsamples[nn[0]] - TTsamples[nn[1]]
              # Generate a random number between 0 and 1
              gap = random.uniform(0, 1)
              # Put the synthesized new sample into the array self.synthetic
              self.synthetic[self.newindex] = TTsamples[nn[1]] + gap * diff

          # Increment self.newindex by 1 to indicate that one more sample has been synthesized
          self.newindex += 1
                
# DSAOC Oversampling Function***********************************************************************************************************************************************
def NOLA_SMOTE(X_train, y_train, X_test, y_test):
    cnum = X_train.shape[1]

    # Calculate the total number of points to be interpolated
    label1_count = y_train.value_counts()[1]  # Count the quantities of the two classes
    label2_count = y_train.value_counts()[0]
    OlaNum = abs(label1_count - label2_count)  # Calculate the total number of interpolation points
    iaper = round(label1_count * 100 / label2_count)

    # Divide the training set into minority class set and majority class set
    ma_data = X_train[y_train == 0]  # Majority class set
    mi_data = X_train[y_train == 1]  # Minority class set

    # Calculate the distance matrices between the minority class points, and between the minority class points and majority class points
    distanceMartix = []
    distanceMartixA = []
    mi_data1 = mi_data.to_numpy()
    ma_data1 = ma_data.to_numpy()

    # Calculate the distance matrix
    for i in range(len(mi_data1)):
        tempdistances_l = [getDistanceByEuclid(mi_data1[i], mi_data1[j]) for j in range(len(mi_data1))]
        distanceMartix.append(tempdistances_l)
        tempdistances_lA = [getDistanceByEuclid(mi_data1[i], ma_data1[j]) for j in range(len(ma_data1))]
        distanceMartixA.append(tempdistances_lA)
    distanceMartix = np.array(distanceMartix)
    distanceMartixA = np.array(distanceMartixA)
    
    # Calculate the scale
    distances = squareform(pdist(X_train))
    sorted_distances = np.sort(distances.ravel())
    # Calculate the number of smaller and larger scales
    percentile_s = np.percentile(distanceMartixA, Bs)
    percentile_b = np.percentile(distanceMartixA, Bg)
    #print("Percentage s = {}".format(percentile_s))
    #print("Percentage b = {}".format(percentile_b))

    # Calculate the local density difference of each minority class point based on the two scale thresholds
    # Under the smaller scale threshold
    global dc, m_DC
    resultDc = np.mean(distanceMartix)
    dc = percentile_s
    #print("dc = {}".format(dc))
    result5, result6, result7, result8 = computeDensities(distanceMartix, distanceMartixA, dc)

    MD_local_density_list = result5  # Density difference corresponding to the scale threshold
    MD__local_densityI_list = result6  # Minority class local density corresponding to the scale threshold
    MD__local_density_listX = result7  # Relative local density of the minority class corresponding to the scale threshold
    MD__local_densityA_list = result8  # Majority class local density corresponding to the scale threshold
    
    # Classify points for the first time based on small-scale results
    # Add records to arrays
    # Create an empty one-dimensional array
    
    N_LAB = np.array([])
    N_LAB0 = np.array([])
    N_LAB1 = np.array([])
    for i in range(len(mi_data1)):
        if MD__local_densityI_list[i] <= 1 and MD__local_densityA_list[i] > 0:
            new_record = 0  # No minority class points within the cutoff, only majority class points, 0
        elif MD__local_densityI_list[i] > 1 and MD__local_densityA_list[i] == 0:
            new_record = 2  # No majority class points within the cutoff, only minority class points, 2  
        elif MD__local_densityI_list[i] > 1 and MD__local_densityA_list[i] > 0 and MD__local_density_listX[i] >= 0.2: 
            new_record = 1  # Both types of points are present within the cutoff, 1
        elif MD__local_densityI_list[i] > 1 and MD__local_densityA_list[i] > 0 and MD__local_density_listX[i] < 0.2: 
            new_record = 0  # Both types of points are present within the cutoff, 1        
        elif MD__local_densityI_list[i] == 1 and MD__local_densityA_list[i] == 0: 
            new_record = 3  # Neither type of points are present within the cutoff, 3        
    
        N_LAB0 = np.append(N_LAB0, MD__local_densityI_list[i])
        N_LAB1 = np.append(N_LAB1, MD__local_densityA_list[i])
        N_LAB = np.append(N_LAB, new_record)
             
    # Use numpy.c_ to stack the original arrays and new columns together
    new_midata0 = np.c_[mi_data1, N_LAB0]
    new_midata01 = np.c_[new_midata0, N_LAB1]
    new_midata1 = np.c_[new_midata01, N_LAB]
    
    
    # Under large-scale cutoff
    mdc = percentile_b
    #print("mdc = {}".format(mdc))
    mresult5, mresult6, mresult7, mresult8 = computeDensities(distanceMartix, distanceMartixA, mdc)
    m_DC = mdc
    m_local_density_list = mresult5  # Density difference corresponding to the cutoff
    m__local_densityI_list = mresult6  # Minority class local density corresponding to the cutoff
    m__local_density_listX = mresult7  # Relative local density of minority class corresponding to the cutoff
    m__local_densityA_list = mresult8  # Majority class local density corresponding to the cutoff
    
    
    # Classify minority points based on results under different cutoffs and add to arrays
    # Create an empty one-dimensional array
    N_LAB2 = np.array([])
    N_LAB3 = np.array([])
    N_LAB4 = np.array([])
    N_LAB5 = np.array([])
    N_LAB6 = np.array([])
   
    # Add records to arrays
    for i in range(len(mi_data1)):
        if new_midata1[i][cnum+2] == 0:
            new_record = 0  # Small-scale is 0, noise, 0
        elif new_midata1[i][cnum+2] == 1:
            new_record = 1  # Small-scale is 1, judged as boundary point, 1
        elif new_midata1[i][cnum+2] == 2 and m__local_densityA_list[i] == 0:
            new_record = 4  # Small-scale is 2, no majority class points in large-scale, safe point, 4
        elif new_midata1[i][cnum+2] == 2 and m__local_densityA_list[i] > 0:
            new_record = 2  # Small-scale is 2, majority class points in large-scale, transition point, 2
        elif new_midata1[i][cnum+2] == 3 and m__local_densityI_list[i] == 1 and m__local_densityA_list[i] > 0:
            new_record = 0  # Majority class points in large-scale, no minority class points, noise point, 0
        elif new_midata1[i][cnum+2] == 3 and m__local_densityI_list[i] == 1 and m__local_densityA_list[i] == 0:
            new_record = 4  # No majority class points in large-scale, no minority class points, safe point, 4
        elif new_midata1[i][cnum+2] == 3 and m__local_densityI_list[i] > 1 and m__local_densityA_list[i] > 0:
            new_record = 2  # Minority class points in large-scale and majority class points, transition point, 2
        elif new_midata1[i][cnum+2] == 3 and m__local_densityI_list[i] > 1 and m__local_densityA_list[i] == 0:
            new_record = 4  # Small-scale is 3, minority class points in large-scale, no majority class points, safe point
        else:
            print(new_midata1[i])
            
        N_LAB2 = np.append(N_LAB2, m__local_densityI_list[i])
        N_LAB3 = np.append(N_LAB3, m__local_densityA_list[i])
        N_LAB4 = np.append(N_LAB4, MD__local_density_listX[i])
        N_LAB5 = np.append(N_LAB5, m__local_density_listX[i])
        N_LAB6 = np.append(N_LAB6, new_record)
        
    # Use numpy.c_ to stack the original arrays and new columns together
    global new_midata2
    new_midata22 = np.c_[new_midata1, N_LAB2]
    new_midata23 = np.c_[new_midata22, N_LAB3]
    new_midata24 = np.c_[new_midata23, N_LAB4]
    new_midata25 = np.c_[new_midata24, N_LAB5]
    new_midata2 = np.c_[new_midata25, N_LAB6]
    
    # Column explanations of new_midata2: 0, x-coordinate 1, y-coordinate 2, small-scale minority class local density 3, small-scale majority class local density 4, judgment type of small-scale
    # 5, large-scale minority class local density 6, large-scale majority class local density 7, small-scale minority class relative density 8, large-scale minority class relative density 9, judgment type of large-scale
    
    # Extract different types of points for scatter plot
    datalabel0 = new_midata2[new_midata2[:, cnum+7] == 0]
    datalabel1 = new_midata2[new_midata2[:, cnum+7] == 1]
    datalabel2 = new_midata2[new_midata2[:, cnum+7] == 2]
    datalabel3 = new_midata2[new_midata2[:, cnum+7] == 4]
    
    # Interpolate boundary points and middle points
    # Call N_Smote for interpolation based on rules
    global new_midata3, synthetic_points
    new_midata3 = new_midata2[(new_midata2[:, -1] == 1) | (new_midata2[:, -1] == 2) | (new_midata2[:, -1] == 4)]  # Select middle and border points
    nsmote = N_Smote(N=OlaNum, r=mdc)
    synthetic_points = nsmote.fit(new_midata3, mdc)
    
    # Plot the interpolated results
    # Plot points with different shapes and colors
    plt.scatter(mi_data1[:,0], mi_data1[:,1], c='red', s=15, marker='o', label='minority_points')
    plt.scatter(ma_data1[:,0], ma_data1[:,1], c='green', s=15, marker='^', label='majority_points')
    plt.scatter(synthetic_points[:,0], synthetic_points[:,1], c='blue', s=15, marker='+', label='synthetic_points')
    #plt.legend()
    plt.show()
    
    # Generate oversampled data
    global X_resampled, y_resampled
    synthetic_points1 = synthetic_points[:,:cnum]
    sy_points = pd.DataFrame(synthetic_points1, columns=mi_data.columns)
    sy_points['class'] = 1
    X_resampled = pd.concat([X_train, sy_points.drop(columns='class')])
    y_resampled = pd.concat([y_train, sy_points['class']])
    
    return fenlei(X_resampled, y_resampled, X_test, y_test)

# Main function***************************************************************************************************************************************************************************
if __name__ == '__main__':
    
    # Create an empty DataFrame to store experiment results
    experiment_results = pd.DataFrame()
    
    # Set column names
    column_names = ['MNB_AUC', 'MNB_Gmean', 'MNB_F1', 'LR_AUC', 'LR_Gmean', 'LR_F1',
                    'DTC_AUC', 'DTC_Gmean', 'DTC_F1']
    
    # Create empty columns with column names and add them to the DataFrame
    for column_name in column_names:
        experiment_results[column_name] = []
    
    # Prepare the dataset
    datasetY = pd.read_csv('data7.csv')
    
    # Split the dataset
    global X, y, Bs, Bg
    # Set small scale Bs and large scale Bg
    Bs =5
    Bg =9
    y = datasetY['class']
    X = datasetY.drop(columns='class')
    
    # Create a five-fold stratified cross-validation object
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Repeat the experiment 10 times
    for i in range(10):
        # Perform five-fold cross-validation
        try:
            for train_index, test_index in kfold.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                Danswer = NOLA_SMOTE(X_train, y_train, X_test, y_test)
                Danswer = np.array(Danswer).reshape(1, 9)
                Dw = pd.DataFrame(Danswer, columns=experiment_results.columns)
                experiment_results = pd.concat([experiment_results, Dw])
        except:
            continue
    
    # Calculate the average of each accuracy metric
    final_average = experiment_results.mean(axis=0)
    
    # Print the average of the final results
    print(final_average)
   
 
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   