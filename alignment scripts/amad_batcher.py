import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids


class MiniBatch:

    def __init__(self, options_file, weight_file, layers):
        self.elmo = Elmo(options_file, weight_file, layers)

    def extract(self, sentences, layer):
        # TODO add multithreading/processing
        tensors = []
        batchSize = 128
        batchLoc = 0
        i = 0
        while batchLoc < len(sentences):
           toID = sentences[batchLoc : (batchLoc + batchSize)]
           character_ids = batch_to_ids(toID)
           embeddings = self.elmo(character_ids)
           npTensor = embeddings['elmo_representations'][layer].detach().numpy()
           tensors.append(npTensor)
        return np.asarray(tensors)