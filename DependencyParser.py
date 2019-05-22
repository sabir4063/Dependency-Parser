import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """
            # Placeholders from Config
            maxIteration = Config.max_iter
            batchSize = Config.batch_size
            hiddenSize = Config.hidden_size
            embeddingSize = Config.embedding_size
            learningRate = Config.learning_rate
            displayStep = Config.display_step
            validationStep = Config.validation_step
            numTokens = Config.n_Tokens 
            lambdaParam = Config.lam
            numTransitions = parsing_system.numTransitions()
            flat_dimension = numTokens * embeddingSize
            
            self.train_inputs = tf.placeholder(tf.int32, [None, numTokens])
            self.train_labels = tf.placeholder(tf.int32, [None, numTransitions])
            self.test_inputs = tf.placeholder(tf.int32, [numTokens, ])
            
            mean = 0.0
            stddev = math.sqrt(1.0/parsing_system.numTransitions())
            
            weights_input = tf.Variable(tf.random_normal([hiddenSize, numTokens * embeddingSize], mean=mean, stddev=stddev))
            weights_output = tf.Variable(tf.random_normal([parsing_system.numTransitions(), hiddenSize], mean=mean, stddev=stddev))

            biases_input = tf.Variable(tf.zeros([hiddenSize, 1]))
            self.b2 = tf.Variable(tf.zeros([parsing_system.numTransitions(), 1]))

            train_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embed = tf.reshape(train_embed, [-1, numTokens * embeddingSize])

            self.prediction = self.forward_pass(train_embed, weights_input, biases_input, weights_output)
            self.prediction = tf.transpose(self.prediction)

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction, labels=tf.argmax(tf.transpose(self.train_labels)))
            l2_loss = tf.multiply(lambdaParam, tf.nn.l2_loss(weights_input))
            self.loss = tf.add(self.loss, l2_loss)
            self.loss = tf.reduce_mean(self.loss)

            optimizer = tf.train.GradientDescentOptimizer(learningRate)
            
			# Disable gradient clipping
            #self.app = optimizer.minimize(self.loss)
            
            # Enable gradient clipping
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)
            
            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)
            self.test_pred = tf.transpose(self.test_pred)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print("Initailized")

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print("\nTesting on dev set at step ", step)
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print(result)

        print("Train Finished.")

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print("Starting to predict on test set")
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print("Saved the test results.")
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_input, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        
        # Cube non-linearity
        h = tf.pow(tf.add(tf.matmul(weights_input, tf.transpose(embed)), biases_input), 3)
        #pred = tf.matmul(weights_output, h)
        pred = tf.add(tf.matmul(weights_output, h), self.b2)

        # Sigmoid non-linearity
        #h = tf.nn.sigmoid(tf.add(tf.matmul(weights_input, tf.transpose(embed)), biases_input))
        #pred = tf.matmul(weights_output, h)
        #pred = tf.add(tf.matmul(weights_output, h), self.b2)
        
        # Tanh non-linearity
        #h = tf.nn.tanh(tf.add(tf.matmul(weights_input, tf.transpose(embed)), biases_input))
        #pred = tf.matmul(weights_output, h)
        #pred = tf.add(tf.matmul(weights_output, h), self.b2)

        
        # Relu non-linearity
        #h = tf.nn.relu(tf.add(tf.matmul(weights_input, tf.transpose(embed)), biases_input))
        #pred = tf.add(tf.matmul(weights_output, h), self.b2)
        #pred = tf.matmul(weights_output, h)
        
        # Two layer network(cube,cube)
        """
        #layer - 1
        z1 = tf.add(tf.matmul(weights_input, tf.transpose(embed)), biases_input)
        h1 = tf.pow(z1, 3)
        
        mean = 0.1
        stddev = math.sqrt(1.0 / parsing_system.numTransitions())
        weights_input_2 = tf.Variable(tf.random_normal([Config.hidden_size, Config.hidden_size], mean=mean, stddev=stddev))
        biases_input_2 = tf.Variable(tf.zeros([Config.hidden_size, 1]))
        
        #layer - 2
        z2 = tf.add(tf.matmul(weights_input_2, h1), biases_input_2)
        h2 = tf.pow(z2, 3)
        
        #output layer
        pred = tf.matmul(weights_output, h2)
        """
        
        # Three layer network (cube->relu->tanh)
        """
        #layer - 1
        z1 = tf.add(tf.matmul(weights_input, tf.transpose(embed)), biases_input)
        h1 = tf.pow(z1, 3)
        
        mean = 0.1
        stddev = math.sqrt(1.0 / parsing_system.numTransitions())
        
        weights_input_2 = tf.Variable(tf.random_normal([Config.hidden_size, Config.hidden_size], mean=mean, stddev=stddev))
        biases_input_2 = tf.Variable(tf.zeros([Config.hidden_size, 1]))
        
        #layer - 2
        z2 = tf.add(tf.matmul(weights_input_2, h1), biases_input_2)
        h2 = tf.nn.relu(z2)
        
        
        weights_input_3 = tf.Variable(tf.random_normal([Config.hidden_size, Config.hidden_size], mean=mean, stddev=stddev))
        biases_input_3 = tf.Variable(tf.zeros([Config.hidden_size, 1]))
        
        #layer - 3
        z3 = tf.add(tf.matmul(weights_input_3, h2), biases_input_3)
        h3 = tf.nn.tanh(z3)
        
        #output layer
        pred = tf.matmul(weights_output, h3)
        """
        
        # Three layer network (cube->cube->cube)
        """
        z1 = tf.add(tf.matmul(weights_input, tf.transpose(embed)), biases_input)
        h1 = tf.pow(z1, 3)
        
        mean = 0.1
        stddev = math.sqrt(1.0 / parsing_system.numTransitions())
        
        weights_input_2 = tf.Variable(tf.random_normal([Config.hidden_size, Config.hidden_size], mean=mean, stddev=stddev))
        biases_input_2 = tf.Variable(tf.zeros([Config.hidden_size, 1]))
        
        #layer - 2
        z2 = tf.add(tf.matmul(weights_input_2, h1), biases_input_2)
        h2 = tf.pow(z2, 3)
        
        
        weights_input_3 = tf.Variable(tf.random_normal([Config.hidden_size, Config.hidden_size], mean=mean, stddev=stddev))
        biases_input_3 = tf.Variable(tf.zeros([Config.hidden_size, 1]))
        
        #layer - 3
        z3 = tf.add(tf.matmul(weights_input_3, h2), biases_input_3)
        h3 = tf.pow(z3, 3)
        
        
        #output layer
        pred = tf.matmul(weights_output, h3)
        """
        
        
        return pred


def genDictionaries(sents, trees):
    word = []
    label = []
    pos = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """

    fWord = []
    fPos = []
    fLabel = []
    feature = []

    # Retrieve top three words and POS from the stack
    for j in range(2, -1, -1):
        index = c.getStack(j)
        fWord.append(getWordID(c.getWord(index)))
        fPos.append(getPosID(c.getPOS(index)))

    # Retrieve top three words and POS from the buffer
    for j in range(0, 3, 1):
        index = c.getBuffer(j)
        fWord.append(getWordID(c.getWord(index)))
        fPos.append(getPosID(c.getPOS(index)))

    for j in range(0, 2, 1):
    
    # Retrieve the word/pos/label of the first and second leftmost / rightmost children of the top two words on the stack
        k = c.getStack(j)
        index = c.getLeftChild(k, 1)
        fWord.append(getWordID(c.getWord(index)))
        fPos.append(getPosID(c.getPOS(index)))
        fLabel.append(getLabelID(c.getLabel(index)))

        index = c.getRightChild(k, 1)
        fWord.append(getWordID(c.getWord(index)))
        fPos.append(getPosID(c.getPOS(index)))
        fLabel.append(getLabelID(c.getLabel(index)))

        index = c.getLeftChild(k, 2)
        fWord.append(getWordID(c.getWord(index)))
        fPos.append(getPosID(c.getPOS(index)))
        fLabel.append(getLabelID(c.getLabel(index)))

        index = c.getRightChild(k, 2)
        fWord.append(getWordID(c.getWord(index)))
        fPos.append(getPosID(c.getPOS(index)))
        fLabel.append(getLabelID(c.getLabel(index)))

    # Retrieve the word/pos/label of the left of the leftmost / right of the right children of the top two words on the stack
        index = c.getLeftChild(c.getLeftChild(k, 1), 1)
        fWord.append(getWordID(c.getWord(index)))
        fPos.append(getPosID(c.getPOS(index)))
        fLabel.append(getLabelID(c.getLabel(index)))

        index = c.getRightChild(c.getRightChild(k, 1), 1)
        fWord.append(getWordID(c.getWord(index)))
        fPos.append(getPosID(c.getPOS(index)))
        fLabel.append(getLabelID(c.getLabel(index)))


    feature.extend(fWord)
    feature.extend(fPos)
    feature.extend(fLabel)
    
    return feature


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print(i, label)
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    with open(filename, 'rb') as file:
        dictionary, word_embeds = pickle.load(file, encoding='latin1')
	
    #dictionary, word_embeds = pickle.load(open(filename, 'r'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = list(wordDict.keys())
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print("Found embeddings: ", foundEmbed, "/", len(knownWords))

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    
    for idx in np.argsort(list(labelDict.values())):
        labelInfo.append(list(labelDict.keys())[idx])
    
    parsing_system = ParsingSystem(labelInfo[1:])
    print(parsing_system.rootLabel)

    print("Generating Traning Examples")
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print("Done.")

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

