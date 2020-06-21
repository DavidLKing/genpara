import pdb
import pickle
import sys

pkl = pickle.load(open(sys.argv[1], 'rb'))

for elem in pkl:
    print(elem)

pdb.set_trace()