for file in elmo_singular_swap.tsv gold_singular_swap.tsv ../data/autParaphrasesNoPOS.tsv ../data/autParaphrasesPOS.tsv
do ./score.local.py ../data/GoogleNews-vectors-negative300.bin ../data/glove.6B.300d.txt.word2vec  $file ../data/corrected.tsv ../data/gigaword4.5g.kenlm.bin && mv scored.tsv $file.scored
done
