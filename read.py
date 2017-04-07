import numpy as np
import csv

def readData(wf_name, wf_idx, resolution, year):
#     get the wind turbine parameters from NREL west wind dataset
#     each csv file contains a 10-min wind output in a year with 30MW
#     input:
#         wf_name: the cell matrix for wind farm info, nLocation*1
#                  matrix, each cell contains the wf name list in csv
#         wf_idx: the matrix store the numbered folder contans csv
#         year: if year is 2006 or 2005, have 52560 measurements
#         resolution: the resolution within one hour
#     output:
#         speed: averaged wind speed among local area, nLocation*1 cell in
#                desired resolution
#         gen: total wind power generation among local area, nLocation*1 cell 
#              in desired resolution
#         wind_param: the output wind csv data file in the cell format, each
#                     cell is one Location, contains n turbine
#         capacity: the capacity of each wind farm, nLocation*1 matrix
#  data:
#       load data under the local folder '2006/'
#  ========================================================================

    nLocation = len(wf_name) # number of sites

    if year % 4 == 0:        # number of measurements
        nRow = 6*24*366      # lunar year
    else:
        nRow = 6*24*365
    
    # initialization
    wind_param = []
    speed_temp = []
    speed = []
    gen = []
    gen_temp = []
    capacity = []
    
    for iLocation in range(nLocation):
        wf_id = wf_idx[iLocation] # the name (number) of iLocation in RTS
        farm_idx = wf_name[iLocation] # pick wind sites in ith Location
        nSite = len(farm_idx) # number of sites in iLocation
        turbine_param = np.zeros((nSite, nRow, 4)) # parameters in each farm
        
        # copy from csv files into the matrices
        for iSite in range(nSite):
            with open('./2006/2006/' + str(wf_id) + '/' + str(farm_idx[iSite]) + '.csv') as f:
                reader = csv.reader(f)
                next(reader)
                count = 0
                for row in reader:
                    turbine_param[iSite, count, :] = row[1:]
                    count += 1  
        # capacity
        loc_capacity = 30*nSite
        capacity.append(loc_capacity)            
        wind_param.append(turbine_param)
        speed_temp.append(np.mean(turbine_param[:,:,0],axis=0))
        gen_temp.append(np.sum(turbine_param[:,:,3], axis=0)/(loc_capacity))
        
        # 1-hr resolution
        if resolution == 1:
            speed_per_hour = np.reshape(speed_temp[iLocation], (nRow//6, 6))
            gen_per_hour = np.reshape(gen_temp[iLocation], (nRow//6, 6))
            speed.append(np.mean(speed_per_hour, axis=1)/30)
            gen.append(np.mean(gen_per_hour, axis=1)) 
        # 10-min resolution
        elif resolution == 6:
            speed.append(speed_temp[iLocation]/30)
            gen.append(gen_temp[iLocation])
        else:
            print ("desired resolution is not valid.")
        
    speed = np.array(speed)
    gen = np.array(gen)
    wind_param = np.array(wind_param)
    capacity = np.array(capacity)
    
    return speed, gen, wind_param, capacity