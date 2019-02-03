import math
import numpy as np
from scipy.spatial import distance


#==============================================================================
# TimeSeries class
#==============================================================================

class TimeSeries(object):
    '''
    Class responsible for modeling a time series.
    '''

    # Constants:

    NEGATIVE_INFINITY = float('-inf')
    POSITIVE_INFINITY = float('+inf')

    def __init__(self, data_points):
        self.data_points = np.array(data_points)
        self.mean = None
        self.std_dev = None

    def size(self):
        return len(self.data_points)

    def reduce_dimension(self, w):
        '''

        Transform this time series object in another one which has only fewer
        points, preserving the overall shape. It is known as PAA transformation
        and was proposed by E. Keogh and colleagues. The ideia is to generate w
        values from each time series subsequence of size n. Each w is a mean
        values of n/w values.

        param w is the length of a subsequence in the transformed time series.
        return a Vector of reals representing a reduced time series.

        '''
        # Checking parameters

        # particular for default parameters check purposes
        if type(w) is not int:
            raise TypeError('w is supposed to be an int')
            
        if w > self.size():
            raise ValueError('w is not supposed to be greater than the time\
                             series')

        # operations

        sub_size = int(math.floor(self.size() / w))
        num_of_means = int(math.floor(self.size() / sub_size))
        mean_values = list()

        # The ideia is to generate w values from each time series subsequence 
        # of size n. Each w is a mean values of n/w values

        for i in range(num_of_means):
            current = self.data_points[i * sub_size: (i + 1) * sub_size]

            # Compute the mean value
            # mean = sum(current) / len(current)
            mean = np.mean(current)
            # add the new mean to the array of means
            mean_values.append(mean)

        mean_values = np.array(mean_values)
        reduced = TimeSeries(mean_values)

        return reduced

    def discretize(self, discr_amount):
        '''

        Computes a discretized version of a reduced time series for a given 
        discretization threshold. This approach does not needs an alphabet and,
        therefore, uses implicitly integers as alphabet, although each point is
        represented as a Double.

        '''
        discretized = np.floor_divide(self.data_points, 1/discr_amount)
        return TimeSeries(discretized)
    
    
    def sub_sequence(self, start, end):
        '''
        
        computes the timeseries object corresponding to the datapoints from
        start to end
        start: an int representing the start index to be considered and 
        included
        end: an int representing the end index to be considered but not
        included
        '''
        return TimeSeries(self.data_points[start:end])

    def standardize(self):
        '''

        Normalizes the time series with a given mean and variation, such that
        the new mean is 0 and variation is 1. This method is parametric to
        enable the normalization with non-local datasets, as occurs in a 
        distributed environment.

        returns a normalized time series for given mean and variation.

        '''
        self._compute_mean()
        self._compute_std_dev()

        std_values = list()

        for point in self.data_points:
            std_values.append((point - self.mean) / self.std_dev)

        return TimeSeries(std_values)
        

    def _compute_mean(self):
       self.mean = np.mean(self.data_points)

    def _compute_std_dev(self):
        self.std_dev = np.std(self.data_points)

    def get_mean(self):
        if self.mean is None:
            self._compute_mean()

        return self.mean

    def get_std_dev(self):
        if self.std_dev is None:
            self._compute_std_dev()
        return self.std_dev
    
    def __add__(self, other):
        data_points = np.concatenate((self.data_points, other.data_points),
                                    axis=0)
        return TimeSeries(data_points)

    def __str__(self):
        string = 'DataPoints:\n'

        for point in self.data_points:
            string += str(point) + '\n'

        return string




#==============================================================================
# Distance Functions
#==============================================================================

def euclidean_distance(a, b):
    '''Returns the euclidean distance bewteen a and b, where a and b are
    points in the same multidimensional space represented by numpy arrays '''
    return distance.euclidean(a, b)




#==============================================================================
# I/O
#==============================================================================

class Reader(object):
    """
    Class intended to open a given dataset and create objects so the data can be
    consumed. 
    """

    def __init__(self, file_name):
        """
        Constructor method.
        file_name : a string with the full path of the file to be opened
        """
        self.file_name = file_name
        self.file = open(self.file_name, mode='r')
        self.data_points = []
        for line in self.file.readlines():
            self.data_points.append(float(line.rstrip()))
        self.file.close()
    def create_time_series(self):
        """
        creates a timeseries object from the file
        """
        return TimeSeries(self.data_points)




#==============================================================================
# Kernels
#==============================================================================

def gaussian(x):
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp((-1.0 / 2.0) * math.pow(x, 2))


def inverse_gaussian(y):
    return math.sqrt(-2 * math.log(math.sqrt(2 * math.pi) * y))




#==============================================================================
# Data Structures
#==============================================================================
class MyBkNode(object)
