'''
Helper method for the .ipynb notebook(s) which contain the experiments done with the NYC hotel and taxicab trip data.

author: Dan Saunders (djsaunde@umass.edu)
'''

# imports...
from geopy.distance import vincenty
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

import numpy as np
import matplotlib.pyplot as plt


def get_destinations(pickup_coords, dropoff_coords, hotel_coords, distance, unit):
    '''
    A function which, given the latitude, longitude coordinates, returns an 
    numpy array of (latitude, longitude) pairs such that each pair corresponds 
    to the destination of a taxicab ride orginating from within the distance
    specified in the units specified.
    
    input:
        pickup_coords: tuple of (pickup_lats, pickup_longs)
        dropoff_coords: typle of (dropoff_lats, dropoff_longs)
        hotel_coords: (latitude, longitude) of the hotel we are interested 
            in as the starting point
        distance: a float specifying the distance from the hotel which we 
            consider to be "close enough"
        unit: the unit of the distance parameter
    
    output:
        A numpy array of shape (2, M), where M is equal to the number of 
        taxicab trips which satisfy the distance criterion.
    '''
    
    # define variable to hold taxicab destinations starting from hotel
    destinations = []
    # get number of taxicab trips in dataset N
    N = len(pickup_coords)
    
    print '...getting nearby pickup locations and storing their destinations\n'

    # loop through each pickup long, lat pair
    for idx, pickup in enumerate(pickup_coords):
        
        # print progress to console periodically
        if idx % 100000 == 0:
            print 'progress: (' + str(idx) + ' / ' + str(N) + ')'
        
        # branch based off of unit of distance
        if unit == 'miles':
            # get distance in miles
            cur_dist = vincenty(hotel_coords, pickup).miles
        elif unit == 'meters':
            # get distance in meters
            cur_dist = vincenty(hotel_coords, pickup).meters
        else:
            raise NotImplementedError
                        
        # check for satisfaction of criterion (and throw away big outliers for visualization)  
        if cur_dist <= distance and vincenty(hotel_coords, dropoff_coords[idx]).miles < 50.0:
            # add dropoff coordinates to list if it meets the [unit] [distance] criterion
            destinations.append(dropoff_coords[idx])
            
    print 'progress: (' + str(N) + ' / ' + str(N) + ')\n'
    
    return np.array(destinations).T


def plot_destinations(destinations, append_to_title):
    '''
    Plot destinations latitudes and longitudes to see a "map" of destinations.
    
    input:
        destinations: A numpy array of shape (2, M), where M is equal to the number of taxicab trips which satisfy
        the above distance criterion.
        append_to_title: String to append to the tite of the plot for context.
        
    output:
        Matplotlib plot of datapoints in the range of the latitude, longitude pairs.
    '''

    plt.plot(destinations[0], destinations[1], 'o')
    plt.title('Destinations From ' + append_to_title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    
    
def cv_kmeans(destinations, cluster_range=range(10, 41, 2)):
    '''
    A function to perform K-Means, cross-validating on the number of clusters k.
    
    input:
        destinations: A numpy array of shape (2, M), where M is equal to the number of taxicab trips which satisfy
        the above distance criterion.
        cluster_range: the values of k to consider in the cross-validation.
        
    output:
        The best object score and correspond model, and the best silhouette score and the corresponding model.
    '''
    # create variables for best model and score
    best_score_model, best_silh_model, best_score, best_silh_score = None, None, -np.inf, -np.inf

    # range over values of k, the number of clusters to use, choose the k with the best objective score
    for k in cluster_range:
        # instantiate KMeans object
        model = MiniBatchKMeans(n_clusters=k)

        # fit the model and predict cluster indices of all coordinates
        labels = model.fit_predict(destinations.T)

        # compute score according to K-Means objective
        score = model.score(destinations.T)

        # compute silhouette score metric
        silh_score = silhouette_score(destinations.T, labels, sample_size=10000)

        # if this is the best scoring model so far
        if score > best_score:
            # update the best score and store the model
            best_score, best_score_model = score, model

        # if this is the best silhouette scoring mode so far
        if silh_score > best_silh_score:
            # update the best score and store the model
            best_silh_score, best_silh_model = silh_score, model
    
    return best_score, best_score_model, best_silh_score, best_silh_model


def visualize_clustering(destinations, n_clusters, title):
    '''
    A function which fits a clustering model and plots its visualization in color.
    
    input:
        destinations: A numpy array of shape (2, M), where M is equal to the number of taxicab trips which satisfy
        the above distance criterion.
        n_clusters: The number of clusters to consider in doing K-Means clustering.
        append_to_title: String to append to the tite of the plot for context.
        
    output:
        A visualization of the clusters found by running the K-Means algorithm with the specified number of 
        clusters.
    '''
    # create model, fit it to data, get cluster indices
    model = MiniBatchKMeans(n_clusters=n_clusters)
    labels = model.fit_predict(destinations)
    centroids = model.cluster_centers_

    # plot the data and their cluster labels
    for clstr in range(n_clusters):
        # plot just the points in cluster i
        clstr_points = destinations[ np.where(labels == clstr) ]
        plt.plot(clstr_points[:, 0], clstr_points[:, 1], 'o')
        # plot the centroids
        lines = plt.plot(centroids[clstr, 0], centroids[clstr, 1], 'kx')
        # make the centroid x's bigger
        plt.setp(lines, ms=8.0)
        plt.setp(lines, mew=2.0)

    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

