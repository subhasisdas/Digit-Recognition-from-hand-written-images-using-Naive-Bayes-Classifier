# Entropy is the measure of the disorder in a set of data. 
#The ID3 heuristic uses this concept to come up with the "next best" 
# attribute in the data set to use as a node, or decision criteria, 
# in the decision tree. Thus, the idea behind the ID3 heuristic is to 
# find the attribute that most lowers the entropy for the data set, 
# thereby reducing the amount of information needed to completely describe 
# each piece of data. Thus, by following this heuristic you will essentially 
# be finding the best attribute to classify the records (according to a reduction
# in the amount of information needed to describe the remaining data division)
#  in the data set

import util
import classificationMethod
import math

class Id3(classificationMethod.ClassificationMethod):
 
 def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "id3"
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    self.labels =[]
    self.diction = {}
    
 
 def train(self, trainingData, trainingLabels, validationData, validationLabels):
    print 'training the data:'
#      print self.legalLabels
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    print self.features
    print len(self.features)
    '''Calculating prior table'''
    print 'Evaluating prior list data'
    priorList = []
    priorProbalityList =[]
    lenTrainingLabels = len(trainingLabels)
#     print lenTrainingLabels
    for j in range(10):
        priorList.append(0)
        priorProbalityList.append(0)
    for i in range(lenTrainingLabels):
        t = trainingLabels[i]
        priorList[t]+=1
    for j in range(len(priorList)):
        priorProbalityList[j]= float (priorList[j]/lenTrainingLabels)
    self.priorProbalityList = priorProbalityList
#     print self.priorProbalityList
    print priorList   
    
    print 'Initialising likelihood table'
    likelihoodList = []
    dict = {}
    old = []
    for i in range(len(trainingLabels)):
        if trainingLabels[i] in dict:
#             print 'present'
#             print  trainingLabels[i]
            old = dict[trainingLabels[i]] 
#             print 'old'     
#             print old
            new = trainingData[i].values()
#             print 'new_present'
#             print new
            for j in range(784):
              new[j] += old[j]          
#             print 'new'
#             print new
            dict[trainingLabels[i]] = new
          
        else:
#             print 'adding first time'
#             print trainingLabels[i]
            dict[trainingLabels[i]] = trainingData[i].values()
    print 'here is the dict'
    print dict
    self.diction = dict
    
#     util.raiseNotDefined()
     
 def classify(self, testData):
#     print 'classifying the data'
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.decisionTree(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
    
 def decisionTree(self, datum): 
     guessConuter =util.Counter()
     guessDict ={}
     print 'guessDict'
     print guessDict
     dict = self.diction
     datumList = datum.values()
     print datumList
     index = 0
     for j in range(len(dict.keys())):
       for i in range(len(datumList)):
         myList = []
#          for t in range(len(dict.keys())):
#           myList.append(0)
         if datumList[i]==1:
             
                 if dict[j][i] > 0 :
                     print 'j,i'
                     print j,i
                     myList.append(j)
                     guessDict[j] = myList
                     print myList
                     index = index+1
#                      break
#                  if index == 100:
#                      break
       break
     print 'GuessDict 0 and 99'
     print guessDict
#      print guessDict[99]
     util.raiseNotDefined()