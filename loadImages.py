"""
loads in images for analysis

v. 2021 12 03

"""

# define data directory and image file range
image_directory = '/home/allen/projects/training-data/dataSet'
                     
#############################################################################
#############################################################################

imageDF = pa.loadDir(image_directory)
print("loaded {} images from {}".format(len(imageDF),image_directory))


