import sys
import pdb


for line in open(sys.argv[1], 'r'):
    line = line.strip().split('\t')
    prefix = line[0:5]
    try:
        orig = ' '.join(eval(line[5]))
        # if type(orig) == str:
        #     orig = orig.replace("', '", " ").replace("['", "").replace("']")
    except:
        orig = line[5]
    suffix = line[6:]
    if "[" not in orig:
        print('\t'.join(prefix + [orig] + suffix))

