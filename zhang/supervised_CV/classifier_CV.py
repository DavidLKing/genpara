# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:18:31 2019

@author: 13982
"""
import numpy as np
#import collections
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
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

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class NeuralClassifier:
    
    def __init__(self, info_list, hidden, intermediate_dimension):
        self.model = None
        self.hidden = hidden
        self.intermediate_dimension = intermediate_dimension
        self.dimensions = []
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
        seed = 77
        np.random.seed(seed)
        
        """debug完就删除"""
        print("To DEBUG: ")
        try:
            print(self.input_matrix[:50])
            print(self.output_matrix[:50])
        except:
            print("DEBUGGING FAILED")
        
        
        """👇注意这里的CV用法是否正确👇"""
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        
        for i, (train, test) in enumerate(kfold.split(self.input_matrix, self.output_matrix)):
            print("Fold: " + str(i))
            
            self.model = keras.models.Sequential()
            self.model.add(keras.layers.Dense(self.intermediate_dimension, input_shape=(self.input_size,), activation="sigmoid"))
            """Adding hidden layers to create a neural network"""
            hidden = self.hidden
            
            evaluations = []
            for ii in range(hidden):
                self.model.add(keras.layers.Dense(self.intermediate_dimension, activation="relu", name=str(ii+1)))
                """Batch Normalization - with 'relu' activation"""
                self.model.add(keras.layers.BatchNormalization())
                self.model.add(keras.layers.Dense(1, activation="sigmoid"))
                self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_m])
                
                """keras.callbacks.ModelCheckpoint will automatically save the model after each epoch"""
                callbacks = [EarlyStopping(monitor='acc', min_delta=0, patience=0, verbose=1, restore_best_weights=True),
                             ModelCheckpoint(filepath='pretrained_bert_model_V2_CV.h5', monitor='acc', save_best_only=True, verbose=1)]
                
                self.model.summary()
                self.model.fit(self.input_matrix[train], self.output_matrix[train], epochs=200, batch_size=128, callbacks = callbacks)
                loss, accuracy, f1 = self.model.evaluate(self.input_matrix[test], self.output_matrix[test], verbose=0)
                evaluations.append((loss, accuracy, f1))
            
            avg_loss = 0
            avg_acc = 0
            avg_f1 = 0
            count = 0
            for loss, acc, f1 in evaluations:
                avg_loss += loss
                avg_acc += acc
                avg_f1 += f1
                count += 1
            avg_loss /= count
            avg_acc /= count
            avg_f1 /= count
            
            print("Avg loss: " + str(avg_loss)[:8])
            print("Avg acc: " + str(avg_acc)[:8])
            print("Avg f1: " + str(avg_f1)[:8])
                
                
    """不太确定best model和self.model 到底是否一样"""
    def retrieve_model(self):
        return self.model
        """如果不对就换成下面的"""
#        return keras.models.load_model("pretrained_bert_model.h5")
    
    def retrieve_dimension(self):
        return self.dimension
    
    def get_input_size(self):
        return self.input_size
    
    
    
    


    
    