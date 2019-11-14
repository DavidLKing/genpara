# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:18:31 2019

@author: 13982
"""
import numpy as np
#import collections
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
#import csv

"""Just the cosine distance calculated from Elmo Scores"""
def getVector(line, which='left'):
    if which == 'left':
        return line[4]
    elif which == 'right':
        return line[5]

"""Get the distance of indices of aligned pairs normalized with the longer of the sentence length"""
def getIndexDistanceNormalized(line):
    aligns = line[0]
    sent_len = int(line[7])
    if '====' in aligns or '----' in aligns:
        align = ''
        if '====' in aligns:
            align = aligns.split('====')
        elif '----' in aligns:
            align = aligns.split('----')
        left = align[0][1:-1].split(', ')
        left = [int(i) for i in left]
        right = align[1][1:-1].split(', ')
        right = [int(i) for i in right]
        lsum = np.mean(left)
        rsum = np.mean(right)
        return abs(lsum - rsum) / sent_len
        
    else:
        align = aligns.split('-')
        left = int(align[0])
        right = int(align[1])
        return abs(left - right) / sent_len

"""Get the span of alignment candidate for each sentence"""
def getAlignmentSpan(line, which='left'):
    aligns = line[0]
    sent_len = int(line[7])
    if '====' in aligns or '----' in aligns:
        align = ''
        if '====' in aligns:
            align = aligns.split('====')
        elif '----' in aligns:
            align = aligns.split('----')
        
        ind = []
        if which == 'left':
            left = align[0][1:-1].split(', ')
            ind = [int(i) for i in left]
        elif which == 'right':
            right = align[1][1:-1].split(', ')
            ind = [int(i) for i in right]
        
        max_ind = max(ind)
        min_ind = min(ind)
        return (max_ind - min_ind) / sent_len
    
    else:
        return 0

"""Check if the core of the aligned pairs have the same dependency"""
def hasSameDep(line):
    pass

"""Get the gold label, changes to bool value due to implementation change"""
def getGold(line):
    if line[6] == True:
        return 1
    elif line[6] == False:
        return 0
    return None

"""????SEEMS TO WORK: Feed all vector values to the input matrix for certain example_index"""
def feedVectorToInput(input_matrix, vector, example_index, dimension, which='left'):    
    for i in range(0, len(vector)):
        if which == 'left':
            input_matrix[example_index][i] = vector[i]
        elif which == 'right':
            input_matrix[example_index][dimension+i] = vector[i]

class NeuralClassifier:
    
    def __init__(self, info_list, hidden, intermediate_dimension, poss_sure):
        self.model = None
        self.hidden = hidden
        self.intermediate_dimension = intermediate_dimension
        self.dimensions = []
        if poss_sure == "SURE":
            self.model_name = "pretrained_bert_model_V2_SURE.h5"
        else:
            self.model_name = "pretrained_bert_model_V2_POSS.h5"
        example_index = 0
        
        for line in info_list:
            
            left_vector = getVector(line, which='left')
            right_vector = getVector(line, which='right')
            self.dimensions.append(len(list(left_vector)))
            self.dimensions.append(len(list(right_vector)))
            
            example_index += 1
            
        """以下做法可能耗时过长，看能不能大整改"""
        
        """Assuming the same dimension for every token though"""
        self.dimension = max(self.dimensions)
        """Extra Feature: Any feature beyond two embeddings"""
        num_extra_features = 0
        self.input_size = (2 * self.dimension) + num_extra_features
        
        self.input_matrix = np.zeros((example_index, self.input_size))
        self.output_matrix = np.zeros((example_index, 1))
        
        example_index = 0
        for line in info_list:
            left_vector = getVector(line, which='left')
            right_vector = getVector(line, which='right')
            
#            sent_distance = getIndexDistanceNormalized(line)
            
            feedVectorToInput(self.input_matrix, left_vector, example_index, self.dimension, which='left')
            feedVectorToInput(self.input_matrix, right_vector, example_index, self.dimension, which='right')
            
            gold_label = getGold(line)
            
            self.output_matrix[example_index][0] = gold_label
            
            example_index += 1
        
    
    def train(self):
        """Set a random seed for reproducibility"""
        np.random.seed(77)
        
        """debug完就删除"""
        print("To DEBUG: ")
        try:
            print(self.input_matrix[:50])
            print(self.output_matrix[:50])
        except:
            print("DEBUGGING FAILED")
            
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.intermediate_dimension, input_shape=(self.input_size,), activation="sigmoid"))
        """Adding hidden layers to create a neural network"""
        hidden = self.hidden
        
        #last_hidden_layer_model = None
        for ii in range(hidden):
            self.model.add(keras.layers.Dense(self.intermediate_dimension, activation="relu", name=str(ii+1)))
            """Batch Normalization - with 'relu' activation"""
            self.model.add(keras.layers.BatchNormalization())
            #    if ii+1 == hidden:
            #        """Get the last hidden layer from the name as defined above"""
            #        last_layer_name = str(hidden)
            #        last_hidden_layer_model = keras.models.Model(inputs=model.input,
            #                                 outputs=model.get_layer(last_layer_name).output)
        self.model.add(keras.layers.Dense(1, activation="sigmoid"))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        """keras.callbacks.ModelCheckpoint will automatically save the model after each epoch"""
        callbacks = [EarlyStopping(monitor='acc', min_delta=0, patience=0, verbose=1, restore_best_weights=True),
                     ModelCheckpoint(filepath=self.model_name, monitor='acc', save_best_only=True, verbose=1)]
        
        self.model.summary()
        self.model.fit(self.input_matrix, self.output_matrix, epochs=200, batch_size=128, callbacks = callbacks)
    
    """不太确定best model和self.model 到底是否一样"""
    def retrieve_model(self):
        return self.model
        """如果不对就换成下面的"""
#        return keras.models.load_model("pretrained_bert_model.h5")
    
    def retrieve_dimension(self):
        return self.dimension
    
    def get_input_size(self):
        return self.input_size
    
    
    
    


    
    