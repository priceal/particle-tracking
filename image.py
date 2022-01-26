"""
Loads and displays image. 

VARIABLE SET BY SCRIPT:
image         the loaded image

v. 2021 02 10

"""
imageNumber = 0


##############################################################################
##############################################################################

image = pa.loadim( imageDF['path'][imageNumber] )

print('')
print('displaying image number {}: '.format(imageNumber) + \
      imageDF['path'][imageNumber] )
print('image dimensions: {} '.format(image.shape) )
print( 'image size: {} Mp'.format(image.size/1000000.0))

plt.figure()
pa.showimage( image, imageDF['path'][imageNumber] )

