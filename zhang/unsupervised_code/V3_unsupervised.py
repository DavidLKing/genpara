# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:09:35 2019

@author: 13982
"""
import sys
from V3_classify import UnsupervisedTraining

def main():
    """Only Step↓👇"""
    testing_file = sys.argv[1]
    bert_layer = int(sys.argv[2])
    boundary = float(sys.argv[3])
    """这个boundary前期手动设置，后期可以改成固定范围/值"""
    predicting_data = UnsupervisedTraining(testing_file, bert_layer, boundary)
    print("Start allGold.tsv prediction process")
    predicting_data.predict()
    print("Start allGold.tsv evaluation")
    predicting_data.evaluate()
    
if __name__ == "__main__":
    main()



