import numpy as np

def feature_build(power, speed, para):
#  build the input vector and output from known time series without spatial
#  ===========================================================  
#  skip the first part that don't have the data
#  input features include the most recent power and near future prediction
#  output target is the the actual power output 
#  features are in the recent order, the prediction, the more recent, 
#  and nearby sites the index is smaller
#  input: 
#     power: nLoc*1 cell, each with wind generation, also as input feature
#     speed: nLoc*1 cell, each with wind speed, as input feature
#     para:  parameters to decide the whole model information
#  output:
#     feature:  nLoc*1 cell, each contains m_sample*nFeatures
#     target:   nLoc*1 cell, each contains m_sample*2,col1_true,col2_pred 

    nFarm = para.nFarm

    nDrop = para.drop_length           #  length of dropped data = fea_hist + horizon
    nSample = para.nSample             #  number of whole sample excluding dropped data

    nFeaTotal = para.nFeature          #  total feature length = fea_hist+fea_pred if no space

    nFeaHist = para.fea_hist*para.resolution
    #  feature length for power series (fea_hist)
                                   
    nFeaSpeed = nFeaHist//2             #  feature length for speed series (fea_pred)

    feature = []
    target = []
                            
    # building features   
    for iFarm in range(nFarm):
        fea_temp = np.empty((nSample, nFeaTotal))
        # set up input feature
        for iFea1 in range(nFeaHist):
            # add history as input feature
            fea_temp[:,iFea1] = power[iFarm][nDrop-para.horizon-iFea1 : para.nSeries-para.horizon-iFea1]
        
        for iFea2 in range(nFeaSpeed):
            fea_temp[:,nFeaHist+iFea2] = speed[iFarm][nDrop-iFea2 : para.nSeries-iFea2] 
        
        # set up target output, throw away the drop_length data
        temp = [power[iFarm][nDrop:para.nSeries]]
        target.append(np.transpose(temp))
        feature.append(fea_temp)
    
    feature = np.array(feature)
    target= np.array(target)
        
    return feature, target
