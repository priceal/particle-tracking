# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# create particle grouping for thresholding
print("\ngrouping by particle ...")
trackingGroup = tracking.groupby('particle')

plt.figure()
for name,group in trackingGroup:
    coords = group[['y','x']].to_numpy()
    plt.plot(coords[:,1],coords[:,0],'.-')