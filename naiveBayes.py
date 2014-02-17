# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from __future__ import division
import util
import classificationMethod
import math
import operator
from pip.commands import zip


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    self.diction = {}
    self.priorProbalityList = []
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    
    "*** YOUR CODE HERE ***"
    #idea1
    features_self = self.features
    print 'printing legalabels'
    print self.legalLabels
        
    '''Calculating prior table'''
    print 'Evaluating prior list data'
    priorList = []
    priorProbalityList =[]
    lenTrainingLabels = len(trainingLabels)
    print lenTrainingLabels
    for j in range(10):
        priorList.append(0)
        priorProbalityList.append(0)
    for i in range(lenTrainingLabels):
        t = trainingLabels[i]
        priorList[t]+=1
    for j in range(len(priorList)):
        priorProbalityList[j]= float (priorList[j]/lenTrainingLabels)
    self.priorProbalityList = priorProbalityList
    print self.priorProbalityList
    print priorList   
    
    print 'Initialising likelihood table'
    likelihoodList = []
    dict = {}
    old = []
    for i in range(len(trainingLabels)):
        if trainingLabels[i] in dict:
            old = dict[trainingLabels[i]] 
            new = trainingData[i].values()
            for j in range(784):
              new[j] += old[j]          
            dict[trainingLabels[i]] = new
        else:
            dict[trainingLabels[i]] = trainingData[i].values()         
    for i in range(len(dict.keys())):
        t = dict[i]
        m = priorList[i]
        dict[i] = [round(x/m,2) for x in t]
    self.diction = dict
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    "*** YOUR CODE HERE ***"
    datumList = datum.values()
    tempProbability = []
    for t in range(len(datumList)+1):
        tempProbability.append(0)
    dict = self.diction
    priorProbalityList = self.priorProbalityList
  
    dictProbabilities = {}
    dictMaxProbabilities ={}
    count = 0
    for i in range(len(dict.keys())):
        tempProbability = []
        for t in range(len(datumList)+1):
          tempProbability.append(0)
        count = count+1
        div = 0
        for j in range(len(datumList)):
            if(datumList[j] == 0):
                temp1 = dict[i][j]
                if temp1 ==1 :
                    tempProbability[j] += 1
                else:
                    tempProbability[j] += round((1- temp1),2)
            elif(datumList[j] ==1):
                div = div + 1
                temp2 = dict[i][j]
                if temp2 ==0:
                    tempProbability[j] += 0
                else:
                    tempProbability[j] += round(temp2,2)
        tempProbability[784] = priorProbalityList[i]
        dictProbabilities[i] = tempProbability
        dictMaxProbabilities[i] = (sum(tempProbability)/div)
        logJoint[i] = round(sum(tempProbability)/div,3)
    tt = max(dictMaxProbabilities.iteritems(), key=operator.itemgetter(1))[0]
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    labelProbList=self.diction
    label1ProbList=labelProbList[label1]
    label2ProbList=labelProbList[label2]
    
    label1FeatureProbDict={}
    label2FeatureProbDict={}
    
    for i in range(len(label1ProbList)):
        label1FeatureProbDict[self.features[i]]= label1ProbList[i]
       
    for i in range(len(label2ProbList)):
       label2FeatureProbDict[self.features[i]]= label2ProbList[i]
    
    label1ProbValuesList=label1FeatureProbDict.values() 
    label1ProbKeysList=label1FeatureProbDict.keys() 
    label2ProbValuesList=label2FeatureProbDict.values() 
    label2ProbKeysList=label2FeatureProbDict.keys()
    '''print 'hhghgh test here' 
    print label1ProbList
    print len(label1ProbList)
    print label2ProbList
    print len(label2ProbList)'''
    featureOddsDict={}    
    for i in range(len(label1ProbList)):
        if(label1ProbList[i]==0 or label2ProbList[i]==0):
            continue
        else:
            odds=(label1ProbList[i]/label2ProbList[i])
            featureOddsDict[label1ProbKeysList[i]]=[odds]
    sorted_list=[(k,v) for v,k in sorted([(v,k) for k,v in featureOddsDict.items()],reverse=True)]
    for i in range(100):
        featuresOdds.append(sorted_list[i][0])
    
    print 'number of features returned'
    print len(featuresOdds)
    print featuresOdds    
    return featuresOdds