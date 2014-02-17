'''
Created on 19-Nov-2013

@author: subhasis
"""
Run id3_main.py by the command python id3_main.py data1 test1 
where data1 = training data set
      test1 = testing data set
      
"""
'''

from __future__ import with_statement
from id3Tree import *
import sys
import os.path
import datetime
from sqlite3.dbapi2 import Date

def input_data():
    
    training_filename = sys.argv[1]
    test_filename = sys.argv[2]

    #validating the files
    if (not os.path.isfile(training_filename)) or not os.path.isfile(test_filename):
            print "Error in reading the file(s)"
            sys.exit(0)

    return training_filename, test_filename

def get_attributes(file):
   
    #list the lines in the file
    with open(file, 'r') as f:
        header = f.readline().strip()

    # Parse the attributes separated by comma from the header
    attributes = [attr.strip() for attr in header.split(",")]

    return attributes

def read_data(file, attributes):
   
    with open(file) as f:
        lines = [line.strip() for line in f.readlines()]

    del lines[0]

    # Parse all of the individual data records from the given file
    data = []
    for line in lines:
        data.append(dict(zip(attributes,
                             [datum.strip() for datum in line.split(",")])))
    
    return data
    
def printall(tree, str):
   
    if type(tree) == dict:
        print "\n"
        print "\t%s[%s]                        [Decision]" % (str, tree.keys()[0])
        print "\t=========               \t=========="
        
        for item in tree.values()[0].keys():
            print "%s\t%s" % (str, item)
            printall(tree.values()[0][item], str )
        
    else:
        print "\t%s              ==>          \t%s" % (str, tree)
        print "\n"


if __name__ == "__main__":
    # Get the training and test data filenames from the user
    training_filename, test_filename = input_data()

    #get the attributes and the data set
    attributes = get_attributes(training_filename)
    target_attr = attributes[-1]
    training_data = read_data(training_filename, attributes)
    test_data = read_data(test_filename, attributes)
    
    print str(datetime.datetime.now()),':creating tree'
    
    # Create the decision tree
    decision_tree = create_decision_tree(training_data, attributes, target_attr, gain)

    #print the results
    print str(datetime.datetime.now()),':classifying',"\n"
    
    classification = classify(decision_tree, test_data)

    # Print classification
    print 'Classification :',  
    for item in classification: print item
        
    # Print the tree
    print "\n","Generated Decision Tree","\n"
    
    printall(decision_tree, "")

"""
REFERENCE : for formulating the entropy and gain equations and the training
            dataset, we have referred to 
            http://www.onlamp.com/pub/a/python/2006/02/09/ai_decision_trees.html?
"""