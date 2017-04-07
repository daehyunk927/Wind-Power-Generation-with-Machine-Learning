from collections import namedtuple
import numpy as np

def parameter_gen(x, t, t_scale, t_lead, space_bool):

#  obtain program parameters based on given input
#  ==============================================
#  input:
#     x: a cell array, each cell is a data series
#     t: number of days considering for the feature
#     t_scale: number of points in one hour
#     t_lead:  leading time for prediction = t_horizon
#     space_bool: 1 if space considered, 0 otherwise
#  output:
#     para: a structure indicating many parameters

# ========= the system parameters ==================
    # initialize a structure of parameters called para
    para = namedtuple("para", "nFarm nSeries horizon resolution fea_hist fea_pred fea_type spa_hist spa_pred spa_nloc drop_length nSample nFeature evaluation")
    [nFarm, nSeries] = np.shape(x) # number of wind farms and overall datapoints
    horizon = t_lead               # forecast horizon, lead time
    resolution = t_scale           # hourly data
    
# ========== feature building ====================
    fea_hist = 24*t                # input feature length for history hours before prediction
    fea_pred = 24*t//2     # input feature length for day-ahead predictions
    fea_type = 1                   # number of features type include power and speed
    
    if (space_bool == 0):
        spa_hist = 0                   # input feature length for nearby farm history days
        spa_pred = 0                   # input feature length for nearby farm day-ahead predictions
        spa_nloc = 0                   # number of extra locations builds 
    elif (space_bool == 1):
        spa_hist = 24*t
        spa_pred = 24*t//2
        spa_nloc = 3
        
    drop_length = resolution*fea_hist + horizon 
    # dropped data length
        
    nSample = nSeries-drop_length  # total sample size
    nFeature = ((fea_hist+fea_pred)*fea_type+(spa_hist+spa_pred)*spa_nloc)*resolution
    # total length for each input vector
                                   
# =========== evaluation criteria ==================
    evaluation = 'RMSE'            # evaluation criteria: MAE or RMSE

    p = para(nFarm, nSeries, horizon, resolution, fea_hist, fea_pred, fea_type, spa_hist, spa_pred, spa_nloc, drop_length, nSample, nFeature, evaluation)
    return p
    