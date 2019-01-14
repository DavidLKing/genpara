import pdb

from amad_batcher import MiniBatch

sents = open('gold_singular_swap.src', 'r').readlines()[:256]
print("Loading elmo")
m = MiniBatch('../data/elmo_2x4096_512_2048cnn_2xhighway_options.json', '../data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', 3)
test = m.extract(sents, 2, 128)
pdb.set_trace()