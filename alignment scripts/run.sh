#!/usr/bin/env bash
#./score.sanity.py ../data/GoogleNews-vectors-negative300.bin ../data/glove.6B.300d.txt.word2vec ../data/multiple.cleaned.noarrays.tsv ../data/corrected.tsv ../data/gigaword4.5g.kenlm.bin lexsub
#mv scored.tsv multiple.scored.tsv

#./score.sanity.py ../data/GoogleNews-vectors-negative300.bin ../data/glove.6B.300d.txt.word2vec ../data/lexsub.tsv ../data/corrected.tsv ../data/gigaword4.5g.kenlm.bin lexsub
#mv scored.tsv lexsub.scored.tsv


cut -f 1-40 lexsub.scored.tsv | perl -pe 's/\t/ /g' | grep -v "glove_src_para_sim" > lexsub.feats.tsv
cut -f 1-40 multiple.scored.tsv | perl -pe 's/\t/ /g' | grep -v "glove_src_para_sim" > multiple.feats.tsv

cut -f 52 multiple.scored.tsv | egrep -v "^annot" > multiple.anno.tsv
cut -f 53 lexsub.scored.tsv | egrep -v "^annot"  > lexsub.anno.tsv

paste lexsub.anno.tsv lexsub.feats.tsv > lexsub.ready.tsv
paste multiple.anno.tsv multiple.feats.tsv > multiple.ready.tsv

cat lexsub.ready.tsv multiple.ready.tsv | grep -P -v "^\t.*" > sanity.feats
#cat multiple.ready.tsv | grep -P -v "^\t.*" > sanity.feats
#cat lexsub.ready.tsv | grep -P -v "^\t.*" > sanity.feats

python score_noclass.py > sanity_output.tsv