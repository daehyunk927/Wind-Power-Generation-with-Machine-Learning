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


# WIND FORECAST CORRECTIVE SCENARIOS for 19 wind farms with VARIATION in time_lead
# This script ends with graphical prediction results for three testing days and
# three testing farms but with different time_leads. It also provides
# errors and coefficients of determination for different time_leads. 
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

# list storing names of wind sites in each wind farm     
wf_name = []
for dirname in dataDir:
    temp = [] 
    for filename in os.listdir('./' + str(year) + '/' + str(year) + '/' + dirname):
        temp.append(os.path.splitext(filename)[0])
    wf_name.append(temp)
     
wf_idx = np.array(wf_idx)
wf_name = np.array(wf_name)

# Output cleaned wind speed and power based on the given data
speed, gen, wind_param, capacity = readData(wf_name, wf_idx, resolution, year)

# varied Time Leads
t_lead = [1, 2, 4, 6, 12, 24]
rmse_avg = dict()
mae_avg = dict()
CoDet_avg = dict()
days = [124, 221, 306] # testing days: can be manipulated
farms = [0, 3, 6] # testing farms: can be manipulated
# x-axis with 10 farms
farm_axis = np.arange(10)

# Scikit-Learn commands for multiple algorithms
Estimators = {
                          "Linear Regression": linear_model.LinearRegression(),
                          "Support Vector Machine": svm.LinearSVR(),
                          "Kernel Ridge": kernel_ridge.KernelRidge(),
                          "Random Forest": RandomForestRegressor(),
                          "Gradient Boosting": GradientBoostingRegressor(),
                          "Neural Network": neural_network.MLPRegressor(),
                          "Nearest Neighbor": neighbors.KNeighborsRegressor(), 
}
for name, estimator in Estimators.items():
    rmse_avg[name] = np.empty((len(t_lead), len(days)))
    mae_avg[name] = np.empty((len(t_lead), len(days)))
    CoDet_avg[name] = np.empty((len(t_lead), len(days)))
for t in range(len(t_lead)):
# Load Parameters
    para = parameter_gen(gen, 5, resolution, t_lead[t], 0)
     
    # Build Feature and Target
    feature, target = feature_build(gen, speed, para)

    # Build Training, Validation, and Test sets
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
         
        # dictionary form to store prediction results
        y_test_predict = dict()
        for name, estimator in Estimators.items():
            temprmse = []
            tempmae = []
            tempcoef = []
            temptime = []
            for f in range(len(farm_axis)):
                # fit the training sets
                estimator.fit(xTr[f], yTr[f].reshape(len(yTr[f]),))
                # predict using each algorithm
                y_test_predict[name] = estimator.predict(xTe[f])
                
                # the wind power should be in the range of 0 to 1, so outliers should be taken care of here.
                for h in range(len(y_test_predict[name])):
                    if (y_test_predict[name][h] < 0):
                        y_test_predict[name][h] = 0
                    elif (y_test_predict[name][h] > 1):
                        y_test_predict[name][h] = 1  
                
                # root mean squared error         
                rmse = math.sqrt(np.mean((y_test_predict[name] - yTe[f].reshape(len(yTe[f]),))**2))
                # mean absolute error
                mae = np.mean(abs(y_test_predict[name] - yTe[f].reshape(len(yTe[f]),)))
                temprmse.append(rmse)
                tempmae.append(mae)
                tempcoef.append(estimator.score(xTe[f], yTe[f].reshape(len(yTe[f]),)))
            rmse_avg[name][t][i] = np.mean(temprmse)
            mae_avg[name][t][i] = np.mean(tempmae)
            CoDet_avg[name][t][i] = np.mean(tempcoef)
    
# Visualize the error results for different time leads using MatplotLib                
fig1 = plt.figure()
plt.title("RMSE", fontsize=30)
plt.xlabel("T_Leads", fontsize=20)
plt.ylabel("Values", fontsize=20)
for name, estimator in Estimators.items():
    plt.plot(t_lead, np.mean(rmse_avg[name], axis=1), label=name)
plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.3), 
                 fancybox=True, shadow=True) 
plt.xlim(1,24)
fig2 = plt.figure()
plt.title("MAE", fontsize=30)
plt.xlabel("T_Leads", fontsize=20)
plt.ylabel("Values", fontsize=20)
for name, estimator in Estimators.items():
    plt.plot(t_lead, np.mean(mae_avg[name], axis=1), label=name)
plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.3), 
                 fancybox=True, shadow=True)
plt.xlim(1,24) 
fig3 = plt.figure()
plt.title("Coefficient of Determination", fontsize=30)
plt.xlabel("T_Leads", fontsize=20)
plt.ylabel("Values", fontsize=20)
for name, estimator in Estimators.items():
    plt.plot(t_lead, np.mean(CoDet_avg[name], axis=1), label=name)
plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.3), 
                 fancybox=True, shadow=True)
plt.xlim(1,24)
 
plt.show()
end_time = time.time()
print("Entire Program time: ", end_time - start_time)