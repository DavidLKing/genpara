# Alignment scrips

`build_phrase_table.py` is the core class script which builds
phrase tables once as have them in a standard format (being
\[source, target, annotations\].  You should probably never
call it directly.

`combineGold.py` standardizes annotations from our previous
aligning tool's output.  It can read in multiple files.  Sample
usage: `python combineGold.py *.tsv`

`elmoAlignments.py` just cleans Amad's "No Alignments"
annotations before calling `build_phrase_table.py`. It can
only work on one file at a time.

All scripts currently just write to stdout.

testing... wow I'm going to need to update this a lot
