# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.weights = {}
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            cGrid = [0.001, 0.002, 0.004, 0.008]
        else:
            cGrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, cGrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, cGrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        best_acc = -1  # best accuracy so far on validation set
        cGrid.sort(reverse=True)
        bestParams = cGrid[0]
        best_weights = None
        for c_val in cGrid:
            print("Training for C = ", c_val, "...")
            self.initializeWeightsToZero()
            for iter in range(self.max_iterations): #pass through data i times
                print("Starting MIRA iteration ", iter, "...")

                for datum, label in zip(trainingData, trainingLabels):
                    predicted = self.classify([datum])[0]
                    if predicted == label:
                        continue
                    else:
                        Tau = min(c_val, ((self.weights[predicted] - self.weights[label]) * datum + 1.0) / (2.0 * (datum * datum)))

                        data_copy = datum.copy()
                        for key, val in data_copy.items():
                            data_copy[key] = val * Tau

                        self.weights[label] += data_copy
                        self.weights[predicted] -= data_copy

            #we trained the classifier for this c_val, we test on the validation set
            guesses = self.classify(validationData)

            #check the accuracy
            accuracy = sum([int(guesses[i] == validationLabels[i]) for i in range(len(validationLabels))])

            print("Accuracy for C = ", c_val, " is ", accuracy, "%")

            if accuracy > best_acc or (accuracy == best_acc and c_val < bestParams):
                best_acc = accuracy
                bestParams = c_val
                best_weights = self.weights.copy()


        self.weights = best_weights

        print("finished training. Best cGrid param = ", bestParams)

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """

        weightVector = self.weights[label]
        sortedWeightVector = weightVector.sortedKeys()

        return sortedWeightVector[:100]