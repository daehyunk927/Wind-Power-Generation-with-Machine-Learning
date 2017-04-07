import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from Input.read import readData
from Input.parameter_gen import parameter_gen
from Input.feature_build import feature_build

from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neural_network
from sklearn import kernel_ridge
from sklearn import neighbors

# WIND FORECAST CORRECTIVE SCENARIOS for 19 wind farms
# This script ends with graphical prediction results for 
# three testing days and three testing farms.
#   =================================================================
#   generate wind scenarios based on historic data
#   considering spatial and temporal correlation
#   provide scenarios with better forecast
#   provide scenarios with uncertainty quantification
#   provide reasonable boundary with scenarios
#   combining multiple data mining techniques
#   including Random Forest, SVM, Linear Regression, KNN, NN
#   the data is based on the NREL Western Wind Dataset

# Load Data
start_time = time.time()
year = 2006
resolution = 1  # 1 hr resolution
speed = []
gen = []
# directory where the data is stored
dataDir = os.listdir('./' + str(year) + '/' + str(year))
# number of wind farms in the directory
nLocation = len(dataDir)

# list storing names of wind farms  
wf_idx = []
for dirname in dataDir:
    wf_idx.append(dirname)

nSites = 0 
# list storing names of wind sites in each wind farm 
wf_name = []
for dirname in dataDir:
    temp = [] 
    for filename in os.listdir('./' + str(year) + '/' + str(year) + '/' + dirname):
        temp.append(os.path.splitext(filename)[0])
        nSites = nSites+1
    wf_name.append(temp)
     
wf_idx = np.array(wf_idx)
wf_name = np.array(wf_name)
print(nSites) # 230 in total

# Output cleaned wind speed and power based on the given data
speed, gen, wind_param, capacity = readData(wf_name, wf_idx, resolution, year)
# Load Parameters
para = parameter_gen(gen, 5, resolution, 1, 0)
   
# Build Feature and Target
feature, target = feature_build(gen, speed, para)
print(np.shape(feature[0]))
print(np.shape(target[0]))
   
# Build Training and Test sets
days = [124, 221, 306] # testing days: can be manipulated
farms = [0, 3, 6] # testing farms: can be manipulated
farm_axis = np.arange(nLocation)
  
for f in range(len(farms)):
    fig = plt.figure()
    for i in range(len(days)):
        # prediction hours: 7 days
        test_hour = np.arange((days[i]-1) * 24, (days[i]+6) * 24) - para.drop_length
        test_time = np.transpose(test_hour)
        train_length = 2160 # length of training sets
       
        nFarm = nLocation
        xTr = []
        yTr = []
        xTe = []
        yTe = []
        
        # build training and testing sets here
        for iFarm in range(nFarm):
            xTr1 = feature[iFarm][test_time[0]-train_length : test_time[0]]
            yTr1 = target[iFarm][test_time[0]-train_length : test_time[0]]
            xTe1 = feature[iFarm][test_time[0]:test_time[len(test_time)-1]+1]
            yTe1 = target[iFarm][test_time[0]:test_time[len(test_time)-1]+1]
          
            xTr.append(xTr1)
            yTr.append(yTr1)
            xTe.append(xTe1)
            yTe.append(yTe1)
           
        xTr = np.array(xTr)
        yTr = np.array(yTr)
        xTe = np.array(xTe)
        yTe = np.array(yTe)
          
        print(np.shape(xTr[0]))
        print(np.shape(yTr[0]))
        print(np.shape(xTe[0]))
        print(np.shape(yTe[0]))
        
        # Scikit-Learn commands for multiple algorithms
        Estimators = {
                    "Linear Regression": linear_model.LinearRegression(),
                    "Support Vector Machine": svm.LinearSVR(),
                    "Kernel Ridge": kernel_ridge.KernelRidge(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Neural Network": neural_network.MLPRegressor(),
                    "Nearest Neighbor": neighbors.KNeighborsRegressor() 
        }
        
        # dictionary form to store prediction results
        y_test_predict = dict()
          
        for name, estimator in Estimators.items():
            t1 = time.time() # for computing time
            print (name, "------")
            # fit the training sets
            estimator.fit(xTr[farms[f]], yTr[farms[f]].reshape(len(yTr[farms[f]]),))
            # predict using each algorithm
            y_test_predict[name] = estimator.predict(xTe[farms[f]])
            
            # the wind power should be in the range of 0 to 1, so outliers should be taken care of here.   
            for h in range(len(y_test_predict[name])):
                if (y_test_predict[name][h] < 0):
                    y_test_predict[name][h] = 0
                elif (y_test_predict[name][h] > 1):
                    y_test_predict[name][h] = 1  
            
            # root mean squared error  
            rmse = math.sqrt(np.mean((y_test_predict[name] - yTe[farms[f]].reshape(len(yTe[farms[f]]),))**2))
            # mean absolute error
            mae = np.mean(abs(y_test_predict[name] - yTe[farms[f]].reshape(len(yTe[farms[f]]),)))
            t2 = time.time()
            # Print the results of the performance of each algorithm
            print ("Coefficient of Determination:", estimator.score(xTe[farms[f]], yTe[farms[f]].reshape(len(yTe[farms[f]]),)))
            print ("Root-Mean-Squared Error:", rmse)
            print ("Mean Absolute Error:", mae)
            print ("Time for each algorithm:", t2-t1)
            print()
        
        # Visualize the prediction results using MatplotLib      
        ax = plt.subplot('%d%d%d' %(len(days),1,i+1))
        for name, estimator in Estimators.items():
            ax.plot(y_test_predict[name], label=name)
  
        ax.plot(yTe[farms[f]], label="Real Data", linestyle='--')
        ax.set_title('Day %d' % days[i], fontsize=15)
        ax.set_xlim(0,167)
        ax.set_ylim(0,1)
    fig.suptitle('Prediction Result for Farm %s' % wf_idx[farms[f]], fontsize=30)
    plt.xlabel('7 days since the requested day (hrs)', fontsize=20)
    plt.ylabel('Power Generated', fontsize=20)
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 1), 
                 fancybox=True, shadow=True)  
plt.show()
end_time = time.time()
print("Entire Program time: ", end_time - start_time)