import pdb
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids

class MiniBatch:

    def __init__(self, options_file, weight_file, layers, device=-1):
        # for cuda, 0 or greater
        # for cpu, -1
        self.device = device
        self.elmo = Elmo(options_file, weight_file, layers)
        if device >= 0:
            self.elmo = self.elmo.cuda(device=self.device)

    def extract(self, sentences, layer, batchSize):
        # TODO add multithreading/processing
        total = len(sentences) // batchSize
        tensors = []
        # batchSize = 128
        batchLoc = 0
        # i = 0
        while batchLoc < total:
            print("On batch", batchLoc + 1, "of", total)
            # print("batchLoc", batchLoc)
            # print("batchLoc * batchSize", batchLoc * batchSize)
            # print("(batchLoc + 1) * batchSize", (batchLoc + 1) * batchSize)
            # print("total sents", len(sentences))
            # pdb.set_trace()
            try:
               toID = sentences[(batchLoc * batchSize) : ((batchLoc + 1) * batchSize)]
            except:
               print("This should be the last batch")
               toID = sentences[(batchLoc * batchSize) : ]
            character_ids = batch_to_ids(toID)
            if self.device >= 0:
                character_ids = character_ids.cuda(device=self.device)
            # embeddings = self.elmo(character_ids.cuda())
            embeddings = self.elmo(character_ids)
            if self.device >= 0:
                npTensor = embeddings['elmo_representations'][layer - 1].detach().cpu().numpy()
            else:
                npTensor = embeddings['elmo_representations'][layer - 1].detach().numpy()
            for tns in npTensor:
                tensors.append(tns)
            batchLoc += 1
        tensors = np.asarray(tensors)
        # pdb.set_trace()
        return tensors