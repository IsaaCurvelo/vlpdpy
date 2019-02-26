import math
import numpy as np
from scipy.spatial import distance
from queue import Queue

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

	def __init__(self, datapoints):
		self.datapoints = np.array(datapoints)
		self.mean = None
		self.std_dev = None

	def size(self):
		return len(self.datapoints)

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
			current = self.datapoints[i * sub_size: (i + 1) * sub_size]

			# Compute the mean value
			# mean = sum(current) / len(current)
			mean = np.mean(current)
			# add the new mean to the array of means
			mean_values.append(mean)

		mean_values = np.array(mean_values)
		reduced = TimeSeries(mean_values)

		return reduced

	def reduce_dimension_isaac(self, paa_size):
		s = self.size()
		if paa_size == s:
			return TimeSeries(self.datapoints)
		elif paa_size > s:
			raise ValueError('w is supposed to be less than timeseries size')
		elif s % paa_size == 0:
			return self.reduce_dimension(paa_size)
		else:
			reduced = []
			for i in range(0, paa_size):
				reduced.append(0.0)

			for i in range(0, paa_size * s):
				idx = int(i / s)
				pos = int(i / paa_size)
				reduced[idx] += self.datapoints[pos]

			for i in range(0, paa_size):
				d = reduced[i] / s
				reduced[i] = d
			return TimeSeries(reduced)

	def discretize(self, discr_amount):
		'''

		Computes a discretized version of a reduced time series for a given 
		discretization threshold. This approach does not needs an alphabet and,
		therefore, uses implicitly integers as alphabet, although each point is
		represented as a Double.

		'''
		discretized = np.floor_divide(self.datapoints, 1/discr_amount)
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
		return TimeSeries(self.datapoints[start:end])

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

		for point in self.datapoints:
			std_values.append((point - self.mean) / self.std_dev)

		return TimeSeries(std_values)
		

	def _compute_mean(self):
	   self.mean = np.mean(self.datapoints)

	def _compute_std_dev(self):
		self.std_dev = np.std(self.datapoints)

	def get_mean(self):
		if self.mean is None:
			self._compute_mean()

		return self.mean

	def get_std_dev(self):
		if self.std_dev is None:
			self._compute_std_dev()
		return self.std_dev
	
	def __add__(self, other):
		datapoints = np.concatenate((self.datapoints, other.datapoints),
									axis=0)
		return TimeSeries(datapoints)

	def __str__(self):
		string = 'DataPoints:\n'

		for point in self.datapoints:
			string += str(point) + '\n'

		return string




#==============================================================================
# Distance Functions
#==============================================================================

def euclidean_distance(a, b):
	'''Returns the euclidean distance bewteen a and b, where a and b are
	points in the same multidimensional space represented by numpy arrays '''
	return distance.euclidean(a, b)

def instance_euclidean_distance(a, b):
	'''Returns the euclidean distance bewteen a and b, where a and b are
	instances of patterns in time series '''
	return distance.euclidean(a['datapoints'], b['datapoints'])




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
		self.datapoints = []
		for line in self.file.readlines():
			self.datapoints.append(float(line.rstrip()))
		self.file.close()
	def create_time_series(self):
		"""
		creates a timeseries object from the file
		"""
		return TimeSeries(self.datapoints)




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
class MyBkNode(object):
	def __init__(self, content):
		self.content = content
		self.child_at_distance = {}

	def add_child_at(self, distance, child):
		self.child_at_distance[distance] = child 

	def get_child_at(self, distance):
		return self.child_at_distance[distance]

class MyBKTree(object):
	def __init__(self, distanceFn, elements = None):
		self.distanceFn = distanceFn
		self.root = None

		if elements is not None:
			for element in elements:
				self.add(element)

	def add(self, element):
		if self.root is None:
			self.root = MyBkNode(element)
		else:
			node = self.root
			inserted = False
			while not inserted:
				d = int(self.distanceFn(node.content, element))
				parent = node
				try:
					node = parent.get_child_at(d)
				except Exception as e:
					node = MyBkNode(element)
					parent.add_child_at(d, node)
					inserted = True
					break


	def query(self, query_object, radius):
		matches = []
		q = Queue()
		q.put(self.root)

		while not q.empty():
			node = q.get()
			element = node.content
			real_distance = self.distanceFn(element, query_object)
			discrete_distance = int(real_distance)
			if real_distance <= radius:
				matches.append((element, real_distance))

			min_search_distance = max(discrete_distance - radius, 0)
			max_search_distance = discrete_distance + radius;

			for search_distance in range(min_search_distance, max_search_distance + 1):
				try:
					child_node = node.get_child_at(search_distance)
					q.put(child_node)
				except Exception as e:
					pass

		return matches




#==============================================================================
# VLPD
#==============================================================================

def is_already_taken(taken, query, radius, dist_fn):
	for p in taken:
		if dist_fn(p, query <= radius):
			return True
	return False

def get_k_denser(sorted_keys, k, radius, dist_fn):
	k_denser = []

	for key in sorted_keys:
		if not is_already_taken(k_denser, key, radius, dist_fn):
			k_denser.append(key)
		
		if len(k_denser) == k:
			break

	return k_denser


def find_patterns(ts, k, n1, n2, c, discretization_amount, w, radius, h):
	print("entering VLPD.findPatterns()..")
	region_size_increment = int((n2 - n1) / (c - 1))
	num_mezzo_breakpoints = c - 2;
	breakpoints = [None] * c
	previous_subsequences = [None] * c
	d = euclidean_distance
	patterns = {}
	all_instances = []
	tree = MyBKTree(instance_euclidean_distance)
	kernel = gaussian

	print("starting to produce and to index subsequences...")
	for t  in range(0, ts.size()):
		# add first breakpoint, corresponding to n1
		breakpoints[0] = t + n1;

		# add the last breakpoint, corresponding to n2
		breakpoints[c - 1] = t + n2

		# add all the intermediate breakpoints
		for i in range(1, num_mezzo_breakpoints + 1):
			breakpoint = t + n1 + i * region_size_increment
			breakpoints[i] = breakpoint
		
		# check if it is possible to generate more subsequences
		if breakpoints[0] > ts.size():
			break

		for i in range(0, c):
			b = breakpoints[i];

			if b > ts.size():
				# can't use a breakpoint greater than the size of the timeseries
				continue


			# generate the subsequence of size b
			sub = ts.sub_sequence(t, b)
			std_sub = sub.standardize()
			
			# reduce with PAA to w points
			rd_sub = std_sub.reduce_dimension_isaac(w)
			disc_sub = rd_sub.discretize(discretization_amount)

			# ---------------------Triviality-check------------------------
			if t > 0:
				s1 = disc_sub.datapoints
				s2 = previous_subsequences[i].datapoints
				if (d(s1, s2) < radius) :
					continue;
			# ------------------end-of-Triviality-check--------------------
			instance = {
				'position': t,
				'size': b - t,
				'datapoints': disc_sub.datapoints
			}

			tree.add(instance)
			all_instances.append(instance)
			previous_subsequences[i] = disc_sub

	print("done...\nbeginning the density calculation process...")

	for instance in all_instances:
		subsequence = tuple(instance['datapoints'])
		if subsequence not in patterns:
			patterns[subsequence] = {
				'representation': subsequence,
				'density': 0.0,
				'instances': list()
			}

		matches = tree.query(instance, radius)
		for match in matches:
			# match is a tuple (instance, distance)
			i = match[0]
			distance = match[1]
			dh = distance / h
			influence = kernel(dh)
			p = patterns[subsequence]
			p['instances'].append(i)
			p['density'] += influence

	print("done...")
	
	sorted_keys = sorted(patterns, key=lambda k: patterns[k]['density'], reverse=True)
	get_k_denser(sorted_keys, k, radius, euclidean_distance)

	k_denser_patterns = [patterns[x] for x in sorted_keys]

	return k_denser_patterns





#==============================================================================
# Test scenario
#==============================================================================
file_name = 'sunspot.dat'

ts = Reader(file_name).create_time_series()


k = 1;
n1 = 100;
n2 = 200;
c = 3;
discretization_amount = 1;
w = 10;
radius = 1;
h = 1.0;

patterns = find_patterns(ts, k, n1, n2, c, discretization_amount, w, radius, h)

for p in patterns:
	print(p)