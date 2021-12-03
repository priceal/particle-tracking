# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:29:52 2021

Particle tracking algorithm that uses coded reference map and dilation
masking method to link particles. Particle detection is through CNN.

VARIABLES SET BY SCRIPT:
tracking     a data frame with tracking results with following columns:
            "particle" : particle number
            "image" :   image number
            "y" :     particle y
            "x" :     particle x
            
survival    a list of numbers of particles in each image
          
v. 2021 12 02

"""

# range of images to process
imageRange = [0,19]

# name of CNN Model to use for particle finding
modelFile = 'CNN_NMPH4_Span2-4_01234.pkl'

# peak finding parameters for final peak filtering of feature map (backend)
minDistance = 4       # min distance between peaks
relThreshold = 0.5    # min relative threshold allowed for peaks

# set parameter for particle linking distance
radius = 3

##############################################################################
# load in the CNN model 
with open(modelFile, 'rb') as file:
    (cnnModel) = pickle.load(file)
    
# create tracking dataframe
tracking = pd.DataFrame ( {"particle" : [],"image" : [], \
                           "y" : [],"x" : [] }, \
                           dtype = int )

# create kernel for dilation of reference images 
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(1+2*radius,1+2*radius))

# find particles in initial image
mapFinal = pa.finderCNN( imageRange[0], cnnModel, imgDF = imageDF )
imageShape = mapFinal.shape
particleCoords = peak_local_max( mapFinal, \
                                 min_distance = minDistance, \
                                 threshold_rel = relThreshold \
                                )
# now create the single particle feature map and dilate
refMap = np.zeros(imageShape,dtype=np.uint16)
refMap[tuple(particleCoords.T)] = np.arange(1,1+len(particleCoords))
refDilated = cv.dilate(refMap,kernel)

# begin main loop
xyList = []; pNumList = []; survival = []
for i in range(imageRange[0],1+imageRange[1]):
    
    # find particles in test image and create feature map
    mapFinal = pa.finderCNN( i, cnnModel, imgDF = imageDF )
    particleCoords = peak_local_max( mapFinal, \
                                 min_distance = minDistance, \
                                 threshold_rel = relThreshold \
                                )
    testMap = np.zeros(imageShape,dtype=np.uint8)
    testMap[tuple(particleCoords.T)] = 1
    
    # mask dilated reference map with test map...
    trackMap = cv.bitwise_and(refDilated,refDilated,mask=testMap)
    
    # ...and store results 
    xyList = np.argwhere( trackMap > 0 ) 
    pNumList = trackMap[ np.nonzero(trackMap) ] 
    imageNum = np.ones((len(pNumList),1),dtype=int)*i
    append_array = np.hstack( (pNumList[:,np.newaxis], imageNum, xyList) )
    append_df = pd.DataFrame(data=append_array,columns=tracking.columns)
    tracking = tracking.append(append_df,ignore_index=True)
   
    # output current number of particles and save in survival list
    print(i,len(pNumList) )
    survival.append( len(pNumList) )

    # now swap in new reference map
    refDilated = cv.dilate(trackMap,kernel)
    
plt.figure()
plt.plot(survival)
print('SUMMARY OF RESULTS: tracking' )
print(tracking.describe())





