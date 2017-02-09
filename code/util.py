from __future__ import division

'''
Helper methods for the .ipynb notebook(s) which contain the experiments done with the NYC hotel and taxicab trip data.

author: Dan Saunders (djsaunde@umass.edu)
'''

# imports...
from geopy.distance import vincenty
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt
import math, itertools, timeit, multiprocessing


# tolerance level for intersection point
EPS = 0.000001

# get number of CPU cores on this machine
num_cores = multiprocessing.cpu_count()


def get_intersection_point(centers, radii):
    '''
    Given four centers and radii, each specifying a circle, calculate their intersection point, if it exists. It should be
    unique.
    
    input:
        centers: coordinates in the (x, y) plane specifying where each of the circles' centers lie
        radii: the length of the circles' radii
        
    output:
        The unique intersection point of the four circles in the (x, y) plane, if it exists. If not, we output None.
    '''
    
    # store x and y dimensions and radius for easy-to-read code 
    x0, x1, x2, x3 = centers[0][0], centers[1][0], centers[2][0], centers[3][0]
    y0, y1, y2, y3 = centers[0][1], centers[1][1], centers[2][1], centers[3][1]
    r0, r1, r2, r3 = radii[0], radii[1], radii[2], radii[3]
    
    # store distances between circle centers (for circle 0 and 1)
    dx = x1 - x0
    dy = y1 - y0
        
    # determine the straight-line distance between the two centers
    dist = math.sqrt(dy ** 2 + dx ** 2)
        
    # check for solutions for the 2-circle case (do these circles intersect or does one contain the other?)
    if dist > r0 + r1:
        return None
    if dist < abs(r0 - r1):
        return None
    
    # calculate distance from the line through the circle intersection points and the line between the circle centers
    a = (r0 ** 2 - r1 ** 2 + dist ** 2) / (2.0 * dist)
    
    # determine coordinates of this point
    point_x = x0 + dx * (a / dist)
    point_y = y0 + dy * (a / dist)
    
    # determine distance from this point to either of the intersection points
    h = math.sqrt(r0 ** 2 - a ** 2)
    
    # determine the offsets of the intersection points from this point
    rx = -dy * (h / dist)
    ry = dx * (h / dist)
    
    # determine the absolute intersection points
    intersection1_x = point_x + rx
    intersection2_x = point_x - rx
    intersection1_y = point_y + ry
    intersection2_y = point_y - ry
    
    # determine if circle 3 intersects at either of the above intersection points
    dx = intersection1_x - x2
    dy = intersection1_y - y2
    d1 = math.sqrt(dy ** 2 + dx ** 2)
    
    dx = intersection2_x - x2
    dy = intersection2_y - y2
    d2 = math.sqrt(dy ** 2 + dx ** 2)
    
    # determine if circle 4 intersects at either of the above intersection points
    dx = intersection1_x - x3
    dy = intersection1_y - y3
    d3 = math.sqrt(dy ** 2 + dx ** 2)
    
    dx = intersection2_x - x3
    dy = intersection2_y - y3
    d4 = math.sqrt(dy ** 2 + dx ** 2)
    
    # check for intersection
    if abs(d1 - r2) < EPS and abs(d3 - r3) < EPS:
        return intersection1_x, intersection1_y
    elif abs(d2 - r2) < EPS and abs(d4 - r3) < EPS:
        return intersection2_x, intersection2_y
    return None

def other_vars_same(prop_row, manhattan_row, capacity):
    '''
    Returns true if all identifying columns are the same. This is to further ensure that we've found the best
    possible match for the hotel identity unmasking.
    
    input:
        prop_row: The first row.
        manhattan_row: The second row.
        capacity: The capacity of the hotel from row prop_row (Came from the "capacities by id" workbook).
        
    output:
        True if all constraints which should be satisfied, are.
    '''
    
    # checking that the rows match on the Operation column
    row1_op, row2_op = prop_row['Operation'], manhattan_row['Operation']
    if not (row1_op == 1 and row2_op == 'Chain Management' or row1_op == 2 and row2_op == 'Franchise' or row1_op == 3 and row2_op == 'Independent'):
        return False

    # checking that the rows match on the Scale column
    row1_scale, row2_scale = prop_row['Scale'], manhattan_row['Scale']
    if not (row1_scale == 1 and row2_scale == 'Luxury Chains' or row1_scale == 2 and row2_scale == 'Upper Upscale Chains' or row1_scale == 3 and row2_scale == 'Upscale Chains' or row1_scale == 4 and row2_scale == 'Upper Midscale Chains' or row1_scale == 5 and row2_scale == 'Midscale Chains' or row1_scale == 6 and row2_scale == 'Economy Chains' or row1_scale == 7 and row2_scale == 'Independents'):
        return False
    
    # checking that the rows match on the Class column
    row1_class, row2_class = prop_row['Class'], manhattan_row['Class']
    if not (row1_class == 1 and row2_class == 'Luxury Class' or row1_class == 2 and row2_class == 'Upper Upscale Class' or row1_class == 3 and row2_class == 'Upscale Class' or row1_class == 4 and row2_class == 'Upper Midscale Class' or row1_class == 5 and row2_class == 'Midscale Class' or row1_class == 6 and row2_class == 'Economy Class'):
        return False
    
    # checking that the rows match on number of rooms / size code columns
    row1_size, row2_size = prop_row['SizeCode'], manhattan_row['Rooms']
    if not (row1_size == 1 and row2_size < 75 or row1_size == 2 and row2_size >= 75 and row2_size <= 149 or row1_size == 3 and row2_size >= 150 and row2_size <= 299 or row1_size == 4 and row2_size >= 300 and row2_size <= 500 or row1_size == 5 and row2_size > 500):
        return False
    
    # checking that the rows match on year opened / date open
    # if not math.isnan(prop_row['OpenDate']) and manhattan_row['Open Date'] != '    -  -  ':
    #    row1_year, row2_year = int(prop_row['OpenDate']), int(manhattan_row['Open Date'][0:4])
    #    if not row1_year == row2_year:
    #        return False
    
    # checking that the capacity of the masked hotel matches the capacity of the hotel from the given row
    # if not int(capacity) == manhattan_row['Rooms']:
    #    return False
    
    # the rows match on all constraints; therefore, return True
    return True

def euclidean_miles(point1, point2):
    '''
    A function which, given two coordinates in UTM, returns the Euclidean distance between the
    two in miles.
    
    input:
        point1: The first point.
        point2: The second point.
        
    output:
        The Euclidean distance between point1 and point2.
    '''
    return math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2)) * 0.000621371

def get_hotel_coords(attr_coords, distances):
    '''
    Function which receives an unordered list of attraction coordinates and corresponding distances, and searches through all
    permutations of these in order to find the correct correspondence (based on whether or not there is an intersection point).
    
    input:
        attr_coords: the latitude, longitude pair for each of the four attractions specified by the data provider
        distances: the distances (in miles) from each attraction to a given hotel
        
    output:
        The coordinates (latitude, longitude) of the hotel in question.
    '''
    
    # try each permutation of the distances
    for perm in itertools.permutations(distances):
        # calculate intersection point
        intersection = get_intersection_point(attr_coords, perm)
        # could come back as NoneType; we check for this here
        if intersection:
            return intersection
        
        
def get_radius(cx, cy, px, py):
    '''
    Calculates the radius of a circle given its center (location of a hotel) and a point on its perimeter.
    
    input:
        cx, cy: (x, y) coordinates of the center
        px, py: (x, y) perimeter coordinates
        
    output:
        The radius of the circle.
    '''
    
    dx = px - cx
    dy = py - cy
    return math.sqrt(dx ** 2 + dy ** 2)


def coords_to_distance_miles(start_coords, end_coords):
    '''
    Given coordinates of two points, calculate the distance between them in miles.
    
    input:
        start_coords: tuple of latitude, longitude coordinates for starting point
        end_coords: tuple of latitude, longitude coordinates for destination point
        
    output:
        The distance between the two points in miles.
    '''
    
    # variables for computing coordinates -> miles
    R = 6371e3
    phi_1, phi_2 = math.radians(start_coords[0]), math.radians(end_coords[0])
    delta_phi = abs(math.radians(start_coords[0] - end_coords[0]))
    delta_lambda = abs(math.radians(start_coords[1] - end_coords[1]))
        
    # computing distance in miles
    a = (math.sin(delta_phi / 2) ** 2) + math.sin(phi_1) * math.sin(phi_2) * (math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def get_destinations3(pickup_coords, dropoff_coords, pickup_times, dropoff_times, passenger_counts, trip_distances, fare_amounts, hotel_coords, distance):
    '''
    A function which, given (latitude, longitude) coordinates, returns an 
    numpy array of (latitude, longitude) pairs such that each pair corresponds 
    to the destination of a taxicab ride orginating from within the distance
    specified in the units specified.
    
    input:
        pickup_coords: tuple of (pickup_lats, pickup_longs)
        dropoff_coords: tuple of (dropoff_lats, dropoff_longs)
        pickup_times: array of datetimes for beginning of taxicab trips
        dropoff_times: array of datetimes for end of taxicab trips
        passenger_counts: the number of passengers per taxicab trip
        trip_distances: the distance in miles per taxicab trip
        fare_amounts: the dollar cost per taxicab trip
        hotel_coords: (latitude, longitude) of the hotel we are interested 
            in as the starting point
        distance: a float specifying the distance from the hotel which we 
            consider to be "close enough"
    
    output:
        A numpy array of shape (2, M), where M is equal to the number of 
        taxicab trips which satisfy the distance criterion.
    '''
    # begin timer
    start_time = timeit.default_timer()
    
    # define variable to hold taxicab destinations starting from hotel
    destinations = []
    
    # pack up inputs
    inputs = zip(pickup_coords, dropoff_coords, pickup_times, dropoff_times, passenger_counts, trip_distances, fare_amounts)
        
    # run 'get_destination()' in parallel for all taxicab trips
    satisfying_trips = Parallel(n_jobs=num_cores)(delayed(get_destination)(*inp, hotel_coords=hotel_coords, distance=distance) for inp in log_progress(inputs, every=10000))
        
    # end timer and report results
    end_time = timeit.default_timer() - start_time
    print '( time elapsed:', end_time, ')', '\n'
    
    return np.array([ trip for trip in satisfying_trips if trip is not None ]).T


def get_destination(pickup, dropoff, pickup_time, dropoff_time, passenger_count, trip_distance, fare_amount, hotel_coords, distance):
    '''
    Helper function for 'get_destinations()'.
    '''
    # get distance in feet
    cur_dist = vincenty(hotel_coords, pickup).feet
                        
    # check for satisfaction of criterion
    if cur_dist <= distance:
        # add dropoff coordinates to list if it meets the [unit] [distance] criterion
        return (round(cur_dist), dropoff[0], dropoff[1], pickup_time, dropoff_time, passenger_count, trip_distance, fare_amounts[idx])
    
    return None


def get_destinations2(pickup_coords, dropoff_coords, pickup_times, dropoff_times, passenger_counts, trip_distances, fare_amounts, hotel_coords, distance, unit):
    '''
    A function which, given (latitude, longitude) coordinates, returns an 
    numpy array of (latitude, longitude) pairs such that each pair corresponds 
    to the destination of a taxicab ride orginating from within the distance
    specified in the units specified.
    
    input:
        pickup_coords: tuple of (pickup_lats, pickup_longs)
        dropoff_coords: tuple of (dropoff_lats, dropoff_longs)
        pickup_times: array of datetimes for beginning of taxicab trips
        dropoff_times: array of datetimes for end of taxicab trips
        passenger_counts: the number of passengers per taxicab trip
        trip_distances: the distance in miles per taxicab trip
        fare_amounts: the dollar cost per taxicab trip
        hotel_coords: (latitude, longitude) of the hotel we are interested 
            in as the starting point
        distance: a float specifying the distance from the hotel which we 
            consider to be "close enough"
        unit: the unit of the distance parameter
    
    output:
        A numpy array of shape (8, M) where M is equal to the number of 
        taxicab trips which satisfy the distance criterion, and we have
        chosen 8 fields of interest to save as relevant data.
    '''
    
    start_time = timeit.default_timer()
    
    # branch based on distance measure
    if unit == 'miles':
        # get distances in miles
        dists = np.array([ vincenty(hotel_coords, pickup).miles for pickup in log_progress(pickup_coords, every=10000) ])
        
    elif unit == 'meters':
        # get distances in meters
        dists = np.array([ vincenty(hotel_coords, pickup).meters for pickup in log_progress(pickup_coords, every=10000) ])
        
    elif unit == 'feet':
        # get distances in feet
        dists = np.array([ vincenty(hotel_coords, pickup).feet for pickup in log_progress(pickup_coords, every=10000) ])
        
    else:
        raise NotImplementedError
                
    # get indices of distances which satisfy the [unit] [distance] criterion
    sat_indices = np.where(dists <= distance)
    
    # end timer and report results
    end_time = timeit.default_timer() - start_time
    
    print '( time elapsed:', end_time, ')', '\n'
    
    # use the satisfying indices to return only the relevant trips
    return np.array([ (dists[idx], dropoff_coords[idx][0], dropoff_coords[idx][1], pickup_times[idx], dropoff_times[idx], passenger_counts[idx], trip_distances[idx], fare_amounts[idx]) for idx in sat_indices ]).T


def get_destinations(pickup_coords, dropoff_coords, pickup_times, dropoff_times, passenger_counts, trip_distances, fare_amounts, hotel_coords, distance, unit):
    '''
    A function which, given (latitude, longitude) coordinates, returns an 
    numpy array of (latitude, longitude) pairs such that each pair corresponds 
    to the starting point of a taxicab ride ending within the distance
    specified in the units specified.
    
    input:
        pickup_coords: tuple of (pickup_lats, pickup_longs)
        dropoff_coords: typle of (dropoff_lats, dropoff_longs)
        hotel_coords: (latitude, longitude) of the hotel we are interested 
            in as the ending point
        distance: a float specifying the distance from the hotel which we 
            consider to be "close enough"
        unit: the unit of the distance parameter
    
    output:
        A numpy array of shape (2, M), where M is equal to the number of 
        taxicab trips which satisfy the distance criterion.
    '''
    
    # begin timer
    start_time = timeit.default_timer()
    
    # define variable to hold taxicab destinations starting from hotel
    destinations = []
    
    # loop through each pickup long, lat pair
    for idx, pickup in log_progress(enumerate(pickup_coords), every=10000, size=len(pickup_coords)):
        
        # branch based off of unit of distance
        if unit == 'miles':
            # get distance in miles
            cur_dist = vincenty(hotel_coords, pickup).miles
        elif unit == 'meters':
            # get distance in meters
            cur_dist = vincenty(hotel_coords, pickup).meters
        elif unit == 'feet':
            # get distance in feet
            cur_dist = vincenty(hotel_coords, pickup).feet
        else:
            raise NotImplementedError
                        
        # check for satisfaction of criterion (and throw away big outliers for visualization)  
        if cur_dist <= distance:
            # add dropoff coordinates to list if it meets the [unit] [distance] criterion
            destinations.append((round(cur_dist), pickup_coords[idx][0], pickup_coords[idx][1], pickup_times[idx], dropoff_times[idx], passenger_counts[idx], trip_distances[idx], fare_amounts[idx]))
            
    # end timer and report results
    end_time = timeit.default_timer() - start_time
    
    print '( time elapsed:', end_time, ')', '\n'
    
    return np.array(destinations).T


def get_starting_points(pickup_coords, dropoff_coords, pickup_times, dropoff_times, passenger_counts, trip_distances, fare_amounts, hotel_coords, distance, unit):
    '''
    A function which, given (latitude, longitude) coordinates, returns an 
    numpy array of (latitude, longitude) pairs such that each pair corresponds 
    to the starting point of a taxicab ride ending within the distance
    specified in the units specified.
    
    input:
        pickup_coords: tuple of (pickup_lats, pickup_longs)
        dropoff_coords: typle of (dropoff_lats, dropoff_longs)
        hotel_coords: (latitude, longitude) of the hotel we are interested 
            in as the ending point
        distance: a float specifying the distance from the hotel which we 
            consider to be "close enough"
        unit: the unit of the distance parameter
    
    output:
        A numpy array of shape (2, M), where M is equal to the number of 
        taxicab trips which satisfy the distance criterion.
    '''
    
    # begin timer
    start_time = timeit.default_timer()
    
    # define variable to hold taxicab destinations starting from hotel
    starting_points = []
    
    # loop through each pickup long, lat pair
    for idx, dropoff in log_progress(enumerate(dropoff_coords), every=10000, size=len(dropoff_coords)):
        
        # branch based off of unit of distance
        if unit == 'miles':
            # get distance in miles
            cur_dist = vincenty(hotel_coords, dropoff).miles
        elif unit == 'meters':
            # get distance in meters
            cur_dist = vincenty(hotel_coords, dropoff).meters
        elif unit == 'feet':
            # get distance in feet
            cur_dist = vincenty(hotel_coords, dropoff).feet
        else:
            raise NotImplementedError
                        
        # check for satisfaction of criterion (and throw away big outliers for visualization)  
        if cur_dist <= distance and vincenty(hotel_coords, dropoff_coords[idx]).miles < 50.0:
            # add dropoff coordinates to list if it meets the [unit] [distance] criterion
            starting_points.append((round(cur_dist), pickup_coords[idx][0], pickup_coords[idx][1], pickup_times[idx], dropoff_times[idx], passenger_counts[idx], trip_distances[idx], fare_amounts[idx]))
            
    # end timer and report results
    end_time = timeit.default_timer() - start_time
    
    print '( time elapsed:', end_time, ')', '\n'
    
    return np.array(starting_points).T


def get_starting_points2(pickup_coords, dropoff_coords, pickup_times, dropoff_times, passenger_counts, trip_distances, fare_amounts, hotel_coords, distance, unit):
    '''
    A function which, given (latitude, longitude) coordinates, returns an 
    numpy array of (latitude, longitude) pairs such that each pair corresponds 
    to the destination of a taxicab ride orginating from within the distance
    specified in the units specified.
    
    input:
        pickup_coords: tuple of (pickup_lats, pickup_longs)
        dropoff_coords: tuple of (dropoff_lats, dropoff_longs)
        pickup_times: array of datetimes for beginning of taxicab trips
        dropoff_times: array of datetimes for end of taxicab trips
        passenger_counts: the number of passengers per taxicab trip
        trip_distances: the distance in miles per taxicab trip
        fare_amounts: the dollar cost per taxicab trip
        hotel_coords: (latitude, longitude) of the hotel we are interested 
            in as the starting point
        distance: a float specifying the distance from the hotel which we 
            consider to be "close enough"
        unit: the unit of the distance parameter
    
    output:
        A numpy array of shape (8, M) where M is equal to the number of 
        taxicab trips which satisfy the distance criterion, and we have
        chosen 8 fields of interest to save as relevant data.
    '''
    
    start_time = timeit.default_timer()
    
    # branch based on distance measure
    if unit == 'miles':
        # get distances in miles
        dists = np.array([ vincenty(hotel_coords, dropoff).miles for dropoff in log_progress(dropoff_coords, every=10000) ])
        
    elif unit == 'meters':
        # get distances in meters
        dists = np.array([ vincenty(hotel_coords, dropoff).meters for dropoff in log_progress(dropoff_coords, every=10000) ])
        
    elif unit == 'feet':
        # get distances in feet
        dists = np.array([ vincenty(hotel_coords, dropoff).feet for dropoff in log_progress(dropoff_coords, every=10000) ])
        
    else:
        raise NotImplementedError
        
    # get indices of distances which satisfy the [unit] [distance] criterion
    sat_indices = np.where(dists <= distance)
    
    # end timer and report results
    end_time = timeit.default_timer() - start_time
    
    print '( time elapsed:', end_time, ')', '\n'
    
    # use the satisfying indices to return only the relevant trips
    return np.array(dists[sat_indices], pickup_coords[sat_indices][0], pickup_coords[sat_indices][1], pickup_times[sat_indices], dropoff_times[sat_indices], passenger_counts[sat_indices], trip_distances[sat_indices], fare_amounts[sat_indices]).T


def log_progress(sequence, every=None, size=None):
    '''
    This method allows me to visualize progress in loops, inline in Jupyter notebooks.
    '''
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{index} / ?'.format(index=index)
                else:
                    progress.value = index
                    label.value = u'{index} / {size}'.format(
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = str(index or '?')


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

