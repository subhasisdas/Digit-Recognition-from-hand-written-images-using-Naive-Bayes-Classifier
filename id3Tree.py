'''
Created on 19-Nov-2013

@author: subhasis
'''
import collections

import math

def entropy(data, target_attr):
    """
    Calculates the entropy : P(+).log(P(+))+P(-).log(P(-))   //log is base2
    """
    dict = {}
    entropy = 0.0

    # for each target attribute, calculate the frequency
    for record in data:
        if (dict.has_key(record[target_attr])):
            dict[record[target_attr]] += 1.0
        else:
            dict[record[target_attr]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for i in dict.values():
        entropy += (-i/len(data)) * math.log(i/len(data), 2) 
        
    return entropy
    
def gain(data, attr, target_attr):
    """
    Calculates the information gain for the attribute attr : Gain(S,A) = E(S) - Summation over Values(A) [ (E(Sv)*|Sv|)/|S| ] 
    """
    dict = {}
    subset_entropy = 0.0

    # for each target attribute, re- calculate the frequency. This is necessary for the gain
    for record in data:
        if (dict.has_key(record[attr])):
            dict[record[attr]] += 1.0
        else:
            dict[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted    
    for val in dict.keys():
        #find probability of occurrence
        div = dict[val] / sum(dict.values())
        setDivider = [record for record in data if record[attr] == val]
        subset_entropy += div * entropy(setDivider, target_attr)

    gain = (entropy(data, target_attr) - subset_entropy)
    return gain
            

def majority_value(data, target_attr):
    data = data[:]
    return most_frequent([record[target_attr] for record in data])

def most_frequent(dataList):
    """
    Returns the item that appears most frequently in the given list.
    """
    dataList = dataList[:]
    highest_freq = 0
    most_freq = None

    for val in pruneNode(dataList):
        if dataList.count(val) > highest_freq:
            most_freq = val
            highest_freq = dataList.count(val)
            
    return most_freq

def pruneNode(dataList):
    """
    Prune the redundant node by removing the repeated values in the dataList
    """
    dataList = dataList[:]
    unique_lst = []

    for item in dataList:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)
            
    # Return the unique list
    return unique_lst

def get_values(data, attr):
   
    data = data[:]
    return pruneNode([record[attr] for record in data])

def choose_attribute(data, attributes, target_attr, fitness):
    """
    Returns the attribute with the highest information gain 
    """
    data = data[:]
    best_gain = 0.0
    best_attr = None

    for attr in attributes:
        gain = fitness(data, attr, target_attr)
        if (gain >= best_gain and attr != target_attr):
            best_gain = gain
            best_attr = attr
                
    return best_attr

def get_matches(data, attr, value):
    """
    Returns a list of all the records in <data> with the value of <attr>
    matching the given value.
    """
    data = data[:]
    rtn_lst = []
    
    if not data:
        return rtn_lst
    else:
        record = data.pop()
        if record[attr] == value:
            rtn_lst.append(record)
            rtn_lst.extend(get_matches(data, attr, value))
            return rtn_lst
        else:
            rtn_lst.extend(get_matches(data, attr, value))
            return rtn_lst

def get_classification(record, tree):
    """
    This function recursively traverses the decision tree and returns a
    classification for the given record.
    """
   #stop condition 1: reached leaf
    if type(tree) == type("string"):
        return tree
    else:
        attr = tree.keys()[0]
        t = tree[attr][record[attr]]
        return get_classification(record, t)

def classify(tree, data):
    """
    Returns a list of classifications for each of the records in the data
    list as determined by the given decision tree.
    """
    data = data[:]
    classification = []
    
    for record in data:
        classification.append(get_classification(record, tree))

    return classification

def create_decision_tree(trainingdata, attributes, target_attr, fitness_func):
    """
    Returns a new decision tree based on training data
    """
    trainingdata = trainingdata[:]
    vals = [record[target_attr] for record in trainingdata]
    default = majority_value(trainingdata, target_attr)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not trainingdata or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = choose_attribute(trainingdata, attributes, target_attr,
                                fitness_func)

        tree = {best:collections.defaultdict(lambda: default)}

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(trainingdata, best):
            subtree = create_decision_tree(
                get_matches(trainingdata, best, val),
                [attr for attr in attributes if attr != best],
                target_attr,
                fitness_func)

            # Add the new subtree to the empty dictionary object in the new
            # tree/node we just created.
            tree[best][val] = subtree

    return tree