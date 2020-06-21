import sys

dial = 0
turn = 0

outfile = open("wilkinsWithDialTurn.csv", 'w')

for line in open(sys.argv[1], 'r'):
    if line.startswith("#START"):
        dial += 1
    else:
        line = line.strip()
        turn += 1
        line = line + "\t({}, {})\n".format(dial -1, turn-1)
        outfile.write(line)