#!/usr/bin/env python
import sys
import cPickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from data import get_data, reload_file
from classifiers import create_classifier, calculate_model_accuracy

__author__ = "Mary Ziemba and Alex Deckey, based on code by Mark Nemececk for COMPSCI 270, Spring 2017, Duke University"
__copyright__ = "Mary Ziemba and Alex Deckey"
__credits__ = ["Mary Ziemba", "Alex Deckey", "David Duquette",
                    "Camila Vargas Restrepo", "Melanie Krassel"]
__license__ = "Creative Commons Attribution-NonCommercial 4.0 International License"
__version__ = "1.0.0"
__email__ = "mtz3@duke.edu"

def classifier(print_option=False):
	'''
	Main Function. Creates a classifier
	'''

	# Create data train/test split
	data_train, data_test, target_train,target_test  = get_data(range(2002,2017),custom=False)

	# Create 2017 test dataset
	data_test_2017, target_test_2017, matchups_2017 = get_data([2017],custom=True)
	
	model_types = ['decision_tree', 'knn', 'gaussian_nb', 'random_forest']
	for model_type in model_types:
		if model_type == 'random_forest':
			f = open('classifier/rf_best_3.pkl', 'rb')
			sys.stdout.flush()
			model = cPickle.load(f)
			print model
		else:
			model = create_classifier(model_type)
		
		# Fit the data to the model
		model.fit(data_train, target_train)

		# Predict using the fit model
		predict_train = predict_with_model(model, data_train)
		predict_test = predict_with_model(model, data_test_2017)
		print; print "=" * 15,; print " Predicting using " + str(model_type) + ' classifier ',; print "=" * 15

		if print_option:
			for matchup,target,predict in zip(matchups_2017,target_test_2017,predict_test):
		 		print str(matchup) + " Actual: " + str(target) + " Predicted: "  + str(predict),
		 		if int(matchup[1]) > int(matchup[3]) and int(target) == 0 or int(matchup[3]) > int(matchup[1]) and int(target) == 1:
		 			print " <-- Upset!",
		 		print
	 		sys.stdout.flush()
		accuracy_train, accuracy_test = calculate_model_accuracy(predict_train, predict_test, target_train, target_test_2017)
		print('Training accuracy: {0:3f}, Accuracy on 2017 Tournament: {1:3f}'.format(accuracy_train, accuracy_test)) 
	print
	sys.stdout.flush()
	return model, predict_train, predict_test, accuracy_train, accuracy_test

def predict_with_model(model,data):
	return model.predict(data)

def split_dataset(data, target, train_size=0.8):
	'''
	Splits the provided data and targets into training and test sets
	'''
	data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=train_size, random_state=0)
	return data_train, data_test, target_train, target_test
		

if __name__ == '__main__':
	if len(sys.argv) < 2:
		raise ValueError('No arguments provided')
	elif sys.argv[1] == 'train_test':
		classifier()
	elif sys.argv[1] == 'bracket17':
		run_custom_bracket()
