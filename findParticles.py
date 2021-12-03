"""
Locates particles in an image or a file using CNN.
The CNN must be defined and trained previously.

VARIABLE SET BY SCRIPT:
particleN    data frame containing the X,Y PIXEL COORDS OF PARTICLES FOUND 
             and the intensities of the peaks, N is the image number

v. 2021 10 28

"""
# image number to use for picking
imageNumber = 150 

# name of CNN Model to Use
modelFile = 'CNN_NMPH4_Span2-4_01234.pkl'

# peak finding parameters for final peak filtering of feature map (backend)
minDistance = 4       # min distance between peaks
relThreshold = 0.5    # min relative threshold allowed for peaks

# for particle filtering added as a pre-filter
prefilter = False
if prefilter:
    sigma = 0.5 # smoothing HWHH
    dims = (5,5) # window size for particle filter kernel
    norm = 1500.0    # normalization for +kernal portion
    
# do not change code below this line
######################################################################
######################################################################
# load in the CNN model 
with open(modelFile, 'rb') as file:
    (cnnModel) = pickle.load(file)

# load in image, applying pre-filter if requested, and scale
print("\nloading image file {} ...".format(imageDF['path'][imageNumber] ))
if prefilter:
    tempImage = pa.loadim( imageDF['path'][imageNumber] )
    imageLoaded = pa.particleFilter(tempImage,sigma,dims,norm=norm)
else:
    imageLoaded = pa.loadim( imageDF['path'][imageNumber] )
print("image dimensions {}".format(imageLoaded.shape))

# Applying CNN and using backend to find peaks
print('finding particles ...', end='')
mapFinal = pa.finderCNN( imageLoaded, cnnModel, imgDF = imageDF )
particleCoords = peak_local_max( mapFinal, \
                                 min_distance = minDistance, \
                                 threshold_rel = relThreshold \
                                )
print(" {} particles identified".format(len(particleCoords)))

# create data and to store and place in data frame. Intensities at peak
# positions are recorded also.
intensities = imageLoaded[ particleCoords[:,0], particleCoords[:,1] ]
tempData = np.concatenate( ( np.fliplr(particleCoords), \
               intensities[:,np.newaxis] ), \
                axis=1 )
output = 'particle' + str(imageNumber)
print( 'creating DataFrame ' + output + ' ...' + '\n')    
exec( output + ' = pd.DataFrame(tempData, columns = ["x", "y", "intensity"] )' )
print('SUMMARY OF RESULTS: ' + output)
print(eval(output).describe())

# create plots for evaluation
pa.showPeaks(imageNumber, eval(output)[['x','y']].to_numpy(), imgDF = imageDF )
eval(output).hist(column='intensity',bins=50)

