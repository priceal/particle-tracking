"""

Creates the particle dataframe that contains the means of the tracking 
data series. A column is added that contains the dwell times of the
particles.

VARIABLE SET BY SCRIPT:
particle     data frame containing results
            "particle" : particle number index
            "image" :   image number
            "y" :     particle y
            "x" :     particle x
            "dwell" :  length of image record of particle (dwell)

v. 2021 12 03

"""
# particle df must contain exactly same particles in same order as tracking df
dwellTracking = tracking   # the tracking dataframe                # threshold level

numbins = 80           # bins for histogram

#############################################################################
#############################################################################

# create particle grouping for thresholding
print("\ngrouping by particle and determining dwells ...")
trackingGroup = dwellTracking.groupby('particle')

# create the data frame and store results
print( 'creating particle dataFrame ' )
particle = trackingGroup.mean()
particle['dwell'] = trackingGroup.agg(len)["image"]
print('SUMMARY OF RESULTS: particle')
print(particle.describe())
plt.figure()
particle['dwell'].hist(bins=numbins)


