# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:09:35 2019

@author: 13982
"""
import sys
from classifier import NeuralClassifier
from get_info import TrainingData
from classify import PredictingData

def main():
    """人生第一次用到class，一定要检查这里对不对"""
    
    """Step 1↓👇"""
    
    training_data = TrainingData(sys.argv[1])
    print("Get bert embeddings")
    training_data.train_bert()
    print("BERT_INFO information retrieving...")
    BERT_INFO = training_data.get_info()
    nlp = training_data.get_nlp()
    
    """Step 2↓👇"""
    
    hidden_layers = int(sys.argv[2])
    intermediate_dimensions = int(sys.argv[3])
    neural_classifier = NeuralClassifier(BERT_INFO, hidden_layers, intermediate_dimensions)
    print("Begin neural network training process")
    neural_classifier.train()
    print("Classifier training session finished")
    dimension = neural_classifier.retrieve_dimension()
    best_model = neural_classifier.retrieve_model()
    
    """Step 3↓👇"""
    
    predicting_data = PredictingData(sys.argv[4], best_model, nlp, dimension)
    print("Start allGold.tsv prediction process")
    predicting_data.predict()
    print("Start allGold.tsv evaluation")
    predicting_data.evaluate()    
    
if __name__ == "__main__":
    main()



