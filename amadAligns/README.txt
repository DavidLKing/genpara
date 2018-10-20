This folder conatains the scripts and results for 3 different methods of alignment.

========
Method 1
========
Description: This is the naive aligner that simply gathers the cosine distance between every element of sentence 1 against every element of sentence 2. It then aligns based on the smallest distance and only aligns each word once.

Usage:
python testAlign.py allGold.tsv > testAlignResults.txt

Results:
With a distance threshold of better than 0.8:
Precision: 0.6853881278538813
Recall: 0.7615423642820903
F1: 0.7214611872146119

========
Method 2
========
Description: This is a bipartite aligner that uses the networkX module to align two bipartite graphs made from each sentence. This is only a mono-word aligner and uses weight based aligning.

Usage:
python bipartiteTest.py allGold.tsv > bipartiteTestResults.txt

Results:
With a distance threshold of better than 0.7:
Precision: 0.5164941785252264
Recall: 0.8102486047691527
F1: 0.630851273948252

========
Method 3
========
Description: This is another bipartite aligner that takes any unaligned words and hyphenates them with their parent if they are Noun phrases. It then re-runs the alignment. This is a multi-word aligner.

Usage:
python bipartiteDepTest.py allGold.tsv > bipartiteDepTestResults.txt

Results:
With a distance threshold of better than 0.7:
Precision: 0.5164941785252264
Recall: 0.8102486047691527
F1: 0.630851273948252