"""
Names: Terry Kusunoki-Martin and Joon Park
Date: 3/21/16
"""
import csv
from sklearn.svm import SVC
from numpy import array
from random import shuffle
from numpy.linalg import norm
from operator import itemgetter
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

class Classifier(object):
	"""Base class for KNN, naive Bayes and SVM classifiers.

	Derived classes are expected to implement the train() and predict() methods.

	The test() and cross_validate() methods can be identical across all
	derived classes and should therefore be implemented here.
	"""
	def __init__(self, data_file):
		"""
		data_file is the name of a file containing comma-separated values
		with one observation per line. All but the last value on each line
		is treated as x, the input to the learned function. The last value
		is treated as y, the output of the learned function.
		The member variables x_data and y_data are lists with corresponding
		indices. The member variable test_start indicates the first index
		that is considered part of the test set; a value of -1 causes the
		whole data set to be used for training.
		"""
		with open(data_file) as f:
			data = [map(float, l) for l in csv.reader(f)]
		self.x_data = [array(d[:-1]) for d in data]
		self.y_data = [d[-1] for d in data]
		self.test_start = len(self.x_data) #use the whole set for training

	def test(self):
		"""Reports the fraction of the test set that is correctly classified.

		All elements of x_data and y_data from test_start to the end of the
		arrays are treated as the test set. The predict() method is run on each
		x_data element of the test set and the result compared to the
		corresponding y_data element. The fraction that are classified correctly
		is tracked and returned as a float.
		"""
		test_data = range(self.test_start, len(self.x_data))
		count = 0
		for test_pt in test_data:
			if self.predict(self.x_data[test_pt]) != self.y_data[test_pt]:
				count += 1
		return float(count)/len(test_data)

	def cross_validation(self, folds=3, params=[]):
		"""Performs k-fold cross validation to select parameters.

		The x_data and y_data arrays are shuffled (preserving their
		correspondence) then divided into %folds subsets. For each parameter
		value, the classifier is trained and tested %folds times, each time,
		the training set is all but one of the subsets, and the test set is
		the one remaining subset. Error rate is averaged across these tests
		to give a score for the parameter value. The parameter value with the
		lowest average error is returned.
		"""
		x_data_shuf = []
		y_data_shuf = []
		elts = len(self.y_data)
		index_shuf = range(elts)
		shuffle(index_shuf)
		for i in index_shuf:
			x_data_shuf.append(self.x_data[i])
			y_data_shuf.append(self.y_data[i])
		

		distance_breakpts = elts/folds
		lowest_error_param = (None, 101) #(parameter, error percent) tuples
		

		for p in params:
			errs = []
			for i in range(folds):
				if i == (folds - 1):#appending to empty list creates a None value
					test_set = x_data_shuf[distance_breakpts*i:]
					self.x_data = x_data_shuf
					self.y_data = y_data_shuf
				else:
					test_set = x_data_shuf[distance_breakpts*i:distance_breakpts*(i+1)]#always move test_set to the end
					self.x_data = x_data_shuf[:distance_breakpts*i] + (x_data_shuf[distance_breakpts*(i+1):]) + (test_set)
					self.y_data = y_data_shuf[:distance_breakpts*i] + (y_data_shuf[distance_breakpts*(i+1):]) + (y_data_shuf[distance_breakpts*i:distance_breakpts*(i+1)])
				self.test_start = len(self.x_data) - len(test_set)
				self.train(p)
				errs.append(self.test())

			avg_error = float(sum(errs))/folds
			print "Average Error for parameter " + str(p) + ": %f" % (avg_error)
			if avg_error < lowest_error_param[1]:
				lowest_error_param = (p, avg_error)

		return lowest_error_param[0]



class NaiveBayes(Classifier):
	def __init__(self, data_file):
		Classifier.__init__(self, data_file)
		self.dict = {}
		self.label_probs = {}

	def train(self, equiv_samples=10):
		"""Computes the probability estimates for naive Bayes.

		Classifying an arbitrary test point requires an estimate of the
		following probabilities:
		- for each label l: P(l)
		- for each input dimension x_i and each value for that dimension:
		    P(x_i = v | l) for each label l.
		These estimates are computed by combining the empirical frequency in
		the data set with a uniform prior.

		equiv_samples: determines how much weight to give to the uniform prior.

		The dimension of the input, the set of values for each input dimension,
		and the set of labels all need to be determined from the data set.
		"""
		indices = {} #dict mapping labels to list of indices (#occurrences is len of list)
		
		for i in range(len(self.y_data)):
			if self.y_data[i] in indices:
				indices[self.y_data[i]].append(i)
			else:
				indices[self.y_data[i]] = [i]

		#initialize label probabilities
		labels = [l for l in indices]
		label_counts = [len(indices[l]) for l in labels]
		for i in range(len(labels)):
			self.label_probs[labels[i]] = label_counts[i]/len(self.y_data)
		
		#use each dimension's index as it's key in the training dict
		num_dimensions = len(self.x_data[0])

		#get all unique values in data
		values = set()
		for row in self.x_data:
			for v in row:
				values.add(v)


		num_vals = len(values)
		for l in indices.keys():
			curr_indices = indices[l]
			count = len(curr_indices)
			self.dict[l] = {}
			for d in range(num_dimensions):
				self.dict[l][d] = {}
				for v in values:
					count_lv = self.count_values(curr_indices, d, v)
					p = (((1.0/num_vals)*(equiv_samples)) + count_lv)/(equiv_samples + count)
					self.dict[l][d][v] = p#store each p(xi=vi|label = li) in dict


	def count_values(self, label_locs, dimension, value):
		count = 0
		for i in label_locs:
			if self.x_data[i][dimension] == value:
				count+=1
		return count

	def predict(self, test_point):
		"""Returns the most probable label for test_point.

		Uses the stored probability of each label and conditional probabilities
		of test_point's input values from self.train().
		"""
		highest_prob = None
		highest_prob_label = None

		for label in self.dict:
			p = self.label_probs[label]#start with label prob and multiply by p(xi=vi|label = li) for each dimension of test pt
			for dim in range(len(test_point)):
				p *= self.dict[label][dim][test_point[dim]]
			if p > highest_prob:
				highest_prob = p
				highest_prob_label = label


		return highest_prob_label


class KNearestNeighbors(Classifier):
	def __init__(self, data_file):
		Classifier.__init__(self, data_file)
		self.k = 0
		self.allLabels = []

	def train(self, k=3):
		"""
		k: number of neighbors considered when classifying a new point.

		Very little setup should be required here!
		"""
		self.k = k
		for i in self.y_data:
			if i not in self.allLabels:
				self.allLabels.append(i)



	def predict(self, test_point):
		"""Returns the plurality class over the k closest training points.

		Nearest neighbors are chosen by Euclidean distance.
		Ties are broken by reducing k by increments of 1 until a strict
		plurality is achieved.
		"""
		allEuclideanDistances = sorted([(norm(test_point - self.x_data[i]), self.y_data[i]) for i in range(self.test_start)], key=itemgetter(0))
		

		tie_flag = True
		cur_k = self.k
		while tie_flag:
			tie_distance = allEuclideanDistances[cur_k - 1][0]
			ties = []
			[ties.append(x) if x[0] == tie_distance else None for x in allEuclideanDistances[cur_k:]]
			KNearest = allEuclideanDistances[:cur_k] + ties

			KNearestLabel = [i[1] for i in KNearest]
			highest_label = ''.join([str(i) for i in self.allLabels])#create label that doesn't exist in the set of labels
			bestCount = 0
			for i in self.allLabels:
				currentCount = KNearestLabel.count(i)
				if currentCount > bestCount:
					highest_label = i
					bestCount = currentCount
					tie_flag = False
					
				elif currentCount == bestCount:
					tie_flag = True
			if tie_flag:#decrease k to break ties
				cur_k -= 1
					
		return highest_label


class SupportVectorMachine(Classifier):
	"""Wrapper for the sklearn.svm.SVC classifier."""
	def __init__(self, data_file):
		Classifier.__init__(self, data_file)
		self.mms = MinMaxScaler()
		self.x_data = self.mms.fit_transform(self.x_data)

	def train(self, kernel="linear"):
		"""
		kernel: one of 'linear', 'poly', 'rbf', or 'sigmoid'.
		"""
		self.svc_model = SVC(kernel=kernel)
		self.svc_model.fit(self.x_data[:self.test_start],
							self.y_data[:self.test_start])

	def predict(self, test_point):
		# SVC.predict takes one or many test points and always returns an array
		return self.svc_model.predict(test_point)[0]


def main():
	nbc = SupportVectorMachine("spambase.data")
	print nbc.cross_validation(3, ['linear'])
	nbc2 = SupportVectorMachine("spambase.data")
	print nbc2.cross_validation(3, ['poly'])	
	nbc3 = SupportVectorMachine("spambase.data")
	print nbc3.cross_validation(3, ['rbf'])
	nbc4 = SupportVectorMachine("spambase.data")
	print nbc4.cross_validation(3, ['sigmoid'])
	



if __name__ == "__main__":
	main()
