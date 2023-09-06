# naiveBayes.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#

import util
import classificationMethod
import math

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

        bestAccuracyCount = -1 # best accuracy so far on validation set

        # Common training - get all counts from training data
        # We only do it once - save computation in tuning smoothing parameter
        commonPrior = util.Counter() # probability over labels
        commonConditionalProb = util.Counter() # Conditional probability of feature feat being 1
                                      # indexed by (feat, label)
        commonCounts = util.Counter() # how many time I have seen feature feat with label y
                                    # whatever inactive or active

        for index in range(len(trainingLabels)):
            datum = trainingData[index]
            label = trainingLabels[index]
            commonPrior[label] += 1
            for feature, value in datum.items():
                commonCounts[(feature, label)] = commonCounts[(feature, label)] + 1
                if value == 1:
                    commonConditionalProb[(feature, label)] = commonConditionalProb[(feature, label)] + 1

        for k in kgrid: # Smoothing parameter tuning loop!
            prior = util.Counter()
            conditionalProb = util.Counter()
            counts = util.Counter()

            # get counts from common training step
            for key, val in commonPrior.items():
                prior[key] += val
            for key, val in commonCounts.items():
                counts[key] += val
            for key, val in commonConditionalProb.items():
                conditionalProb[key] += val

            # smoothing:
            for label in self.legalLabels:
                for feat in self.features:
                    conditionalProb[ (feat, label)] +=  k
                    counts[(feat, label)] +=  2*k # 2 because both value 0 and 1 are smoothed

            # normalizing:
            prior.normalize()
            for x, count in conditionalProb.items(): #computing conditional
                conditionalProb[x] = count * 1.0 / counts[x]

            self.prior = prior
            self.conditionalProb = conditionalProb

            # evaluating performance on validation set
            predictions = self.classify(validationData)
            accuracyCount =  [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)

            print("Performance on validation set for k=%f: (%.1f%%)" % (k, 100.0*accuracyCount/len(validationLabels)))
            if accuracyCount > bestAccuracyCount:
                bestParams = (prior, conditionalProb, k)
                bestAccuracyCount = accuracyCount
            # end of automatic tuning loop
        self.prior, self.conditionalProb, self.k = bestParams

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum using Naive Bayes rule.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        for label in self.legalLabels:
            logPrior = math.log(self.prior[label])
            logJoint[label] = math.log(self.prior[label])
            #print("logPrior: ", logPrior)
            #print("logJoint[label]: ", logJoint[label])
            for feature, data_value in datum.items():
                incr = math.log(self.conditionalProb[feature, label])
                #print("incr: ", incr)
                #print("data_value: ", data_value)
                if data_value > 0:
                    logJoint[label] += incr
                else:
                    logJoint[label] += math.log(1 - self.conditionalProb[feature, label])

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []
        features = []

        "*** YOUR CODE HERE, to populate featureOdds based on above formula. ***"
        for feature in self.features:
            featuresOdds.append(self.conditionalProb[feature, label1] / self.conditionalProb[feature, label2])
            features.append(feature)

        featuresOdds = [feature for _, feature in sorted(zip(featuresOdds, features), key=lambda x: x[0], reverse=True)]

        return featuresOdds[:100]
