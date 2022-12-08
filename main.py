import random #for testing 
import time 
import copy as setCopy # for copying over list of features 
import math # for the distance function in the nearest neighbor algorithm 

#large dataset 106 (contains 40 features)
#small dataset 33 (contains 6 features)
#first column is the class 
#data is in ASCII Text, IEEE standard for 8 place floating numbers 

#code the nearest neighbor classifier 
#use it in a wrapper that does forward selection and backward elimination (both search algorithms)

#the datasets have two classes only and have continuous features 

#make sure nearest neighbor algorithm is working before attempting search algorithms 


# main program 
def main (): 
  print("Welcome to Zinal Patel's Feature Selection Algorithm.")

  # pulling in filename and algorithm from user
  filename = input("Type in the filename of the dataset you want to test." + '\n')
  algorithm = input("Type in the number of the corresponding algorithm you want to run." + '\n' + "1: Forward Selection" + '\n' + "2: Backward Elimination" + '\n')

  # opening file 
  dataset = open(filename, "r")
    
  # reads the first line and returns the number of features (length of first line - 1 because of the class number in the first column)
  numfeatures = len((dataset.readline()).split())-1 
    
  # using readline to read the number of instances or rows and store it 
  # https://pynative.com/python-count-number-of-lines-in-file/
  with open(filename, "r") as dataset:
    k = len(dataset.readline())
  
  numinstances = k
  
  print("This dataset has ",str(numfeatures)," features (not including the class attribute), with ",str(numinstances)," instances." + '\n')

  # storing our dataset into a 2D array as float for ease of use using "list comprehension" method 
  # https://www.geeksforgeeks.org/python-using-2d-arrays-lists-the-right-way/
  datasetarray = [[] for k in range(numinstances)]
  
  #store each instance in dataset as a float in datasetarray 
  with open(filename, "r") as dataset: 
    for k in range (numinstances):
      datasetarray[k] = [float(i) for i in dataset.readline().split()]
      #for i in dataset.readline().split():
        #datasetarray[k] = float(i)

  # start recording time 
  startTime = time.time()

  # running nearest neighbor and search algorithms with all features in dataset 
  all_features_set = [] # initializing empty set
  for k in range (1, numfeatures+1):
    all_features_set.append(k) 

  accuracy = leaveOneOutCrossValidation(datasetarray, all_features_set, numinstances)
  print("Running nearest neighbor with all ", str(numfeatures), " features, using the leaving one out evaluation, I get an accuracy of ", str(accuracy)," %." + '\n')

  # running search to find best features 
  # the forward selection and backward elimination functions are based off of the featuresSearch code on lecture slides 
  print("Beginning search." + '\n')
  if algorithm == "1":
    forwardSelection(datasetarray, numinstances, numfeatures)
  if algorithm == "2":
    backwardElimination(datasetarray, numinstances, numfeatures, accuracy)
  
  # stop recording time 
  endTime = time.time()

  print("The search for this dataset took {} seconds.".format(endTime - startTime) + '\n') 

  return 
  
# for function stub is for testing accuracy function 
# def leaveOneOutCrossValidation (dataset, current_set, feature_to_add):
#  accuracy = random.randint()
#  return accuracy 

# This function estimates the accuracy of our nearest neighbor classifier by using K-fold cross validation 
# K-fold cross validation = divide dataset in K sections, test K times, leaving one instance out to test classifier 
# Separated from the nearest neighbor function to allow easier testing 
def leaveOneOutCrossValidation (dataset, current_set, numinstances):
  number_correctly_classified = 0.0
  for k in range (numinstances):
    # instance used to test classifier 
    instance_one_out = k

    # if index of neighbor found by the nearest neighbor algorithm matches k, it is correct 
    neighborindex = nearestNeighbor (dataset, current_set, numinstances, instance_one_out)
    if dataset[neighborindex][0] == dataset[instance_one_out][0]:
      number_correctly_classified = number_correctly_classified + 1

  # return the accuracy as a percentage     
  accuracy = (number_correctly_classified/numinstances) * 100
  return accuracy 

# This function is the nearest neighbor algorithm or classifier. It calculates the distance between a new data
# point or instance and its neighbor and classifies the new one based on the class of its neighbor. 
def nearestNeighbor (dataset, current_set, numinstances, object_to_classify):
  
  # holds the label/index of the nearest neighbor 
  nearest_neighbor_label = 0
  
  # holds the distance of nearest neighbor 
  nearest_neighbor_distance = float('inf')
  #nearest_neighbor_location = float('inf')
  
  for k in range(numinstances):
    # If the new instance is not equal to the current instance/object we're checking 
    if k != object_to_classify:
      distance = 0
      for i in range(len(current_set)):

        # distance calculation using standard distance function  
        #temp = str(dataset)[k][str(current_set)[i]]
        distance = distance + pow((dataset[k][current_set[i]] - dataset[object_to_classify][current_set[i]]), 2)
      distance = math.sqrt(distance)

      # comparing whether the distance of a new point is smaller than a current distance we have stored (to find nearest neighbor)
      if distance < nearest_neighbor_distance:
        nearest_neighbor_distance = distance
        nearest_neighbor_label = k

  return nearest_neighbor_label


# This function uses the forward selection search algorithm to determine which feature subset has the highest accuracy. 
# Starts off with no features in the array, tests new features, adds ones with highest accuracy.  
# greedy search 
def forwardSelection(dataset, numinstances, numfeatures):
  current_set_of_features = [] # initiatlizing empty set 
  final_set_of_features = []
  best_so_far_accuracy = 0.0
  #feature_to_add_at_this_level = 0

  # local variables use to continue search in case of local maxima 
  #local_feature = 0
  #local_accuracy = 0.0

  # nested loops to search through feature tree 
  for i in range(numfeatures): 
    #local variables use to continue search in case of local maxima 
    local_feature = 0
    local_accuracy = 0.0

    feature_to_add_at_this_level = 0

    for k in range (1, numfeatures+1):
      if k not in current_set_of_features: # only consider adding if it isn't in the subset of features already 
        new_current_set = setCopy.deepcopy(current_set_of_features) # using deep copy to copy our current set of features to a new set 
        new_current_set.append(k)

        # what is the accuracy if we add another feature to our current set? 
        accuracy = leaveOneOutCrossValidation(dataset, new_current_set, numinstances)
        print("\tUsing feature(s) {", str(new_current_set), "} accuracy is ", accuracy, "%." + '\n')

        # if the accuracy is better, let's add this feature to our final set 
        if accuracy > best_so_far_accuracy:
          best_so_far_accuracy = accuracy
          feature_to_add_at_this_level = k 

        # if the accuracy got worse, we should not add it to final set BUT keep testing in case of local maxima 
        if accuracy > local_accuracy:
          local_accuracy = accuracy
          local_feature = k
    
    # if we have a feature with a higher accuracy to add, add it to final set 
    if feature_to_add_at_this_level > 0: 
      current_set_of_features.append(feature_to_add_at_this_level)
      final_set_of_features.append(feature_to_add_at_this_level)
      print("Feature set ", current_set_of_features, " was best, accuracy is ", best_so_far_accuracy, "%." + '\n')
    
    # continue search in case of local maxima (since we are using greedy search)
    else:
      print("(Warning, Accuracy has decreased! Continuing search in case of local maxima.)" + '\n')
      current_set_of_features.append(local_feature)
      print("Feature set ", current_set_of_features, " was best, accuracy is ", best_so_far_accuracy, "%." + '\n')

  print("Finished search!! The best feature subset is ", final_set_of_features, ", which has an accuracy of ", best_so_far_accuracy, "%." + '\n')


# This function uses the backward elimination search algorithm to determine which feature subset has the highest accuracy. 
# Starts off with all features added to the array, tests accuracies one by one, and then removes the features with the lowest accuracies. 
# greedy search   

# We include accuracy parameter to get the value of the argument from when we ran the nearest neighbor algorithm on the full feature subset. 
# We want this because we are going backwards ! 
def backwardElimination(dataset, numinstances, numfeatures, accuracy):
  best_so_far_accuracy = accuracy # taking the accuracy of full feature subset from before 
  feature_to_remove = 0
 
  # copying over the full feature subset into new sets  
  current_set_of_features = list(range(1,numfeatures))
  final_set_of_features = list(range(1, numfeatures))

  # nested loops to search through feature treee
  for i in range (numfeatures):
    #local variables use to continue search in case of local maxima 
    local_feature = 0
    local_accuracy = 0.0

    feature_to_add_at_this_level = 0

    for k in range(1, numfeatures+1):
      if k in current_set_of_features:
        new_current_set = setCopy.deepcopy(current_set_of_features) # using deep copy to copy our current set of features to a new set 
        new_current_set.remove(k) # remove the feature in question 

        # what is the accuracy now when we have removed this feature from our current set? 
        accuracy = leaveOneOutCrossValidation(dataset, new_current_set, numinstances)
        print("\tUsing feature(s) {", str(new_current_set), "} accuracy is ", accuracy, "%." + '\n')

        # if the accuracy is better, let's remove this feature from our final set 
        if accuracy > best_so_far_accuracy:
          best_so_far_accuracy = accuracy
          feature_to_remove = k

        # if the accuracy got worse, lets remove this feature from our final set BUT keep testing in case of local maxima
        if accuracy > local_accuracy:
          local_accuracy = accuracy
          local_feature = k
    
    # if removing a feature increases our end accuracy, remove it 
    if feature_to_remove > 0:
      current_set_of_features.remove(feature_to_remove)
      final_set_of_features.remove(feature_to_remove)
      print("Feature set ", current_set_of_features, " was best, accuracy is ", best_so_far_accuracy, "%." + '\n')

    # continue search in case of local maxima (since we are using greedy search)
    else:
      print("(Warning, Accuracy has decreased! Continuing search in case of local maxima.)" + '\n')
      current_set_of_features.remove(local_feature)
      print("Feature set ", current_set_of_features, " was best, accuracy is ", best_so_far_accuracy, "%." + '\n')
  
  print("Finished search!! The best feature subset is ", final_set_of_features, ", which has an accuracy of ", best_so_far_accuracy, "%." + '\n')

if __name__ == '__main__':
    main()
