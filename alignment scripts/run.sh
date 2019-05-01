#!/usr/bin/env bash
#./score.sanity.py ../data/GoogleNews-vectors-negative300.bin ../data/glove.6B.300d.txt.word2vec ../data/multiple.cleaned.norepeats.tsv ../data/corrected.tsv ../data/gigaword4.5g.kenlm.bin variable noSpecials
#mv scored.tsv multiple.scored.tsv
#cut -f 1-44 multiple.scored.tsv | perl -pe 's/\t/ /g' | grep -v "glove_src_para_sim" > multiple.feats.tsv
#cut -f 52 multiple.scored.tsv | egrep -v "^annot" > multiple.anno.tsv
#paste multiple.anno.tsv multiple.feats.tsv > multiple.ready.tsv
#cat multiple.ready.tsv | grep -P -v "^\t.*" > sanity.feats

#./score.sanity.py ../data/GoogleNews-vectors-negative300.bin ../data/glove.6B.300d.txt.word2vec ../data/lexsub.tsv ../data/corrected.tsv ../data/gigaword4.5g.kenlm.bin lexsub noSpecials
#mv scored.tsv lexsub.scored.tsv
#cut -f 1-44 lexsub.scored.tsv | perl -pe 's/\t/ /g' | grep -v "glove_src_para_sim" > lexsub.feats.tsv
#cut -f 57 lexsub.scored.tsv | egrep -v "^annot"  > lexsub.anno.tsv
#paste lexsub.anno.tsv lexsub.feats.tsv > lexsub.ready.tsv
#cat lexsub.ready.tsv | grep -P -v "^\t.*" > sanity.feats

#./score.sanity.py ../data/GoogleNews-vectors-negative300.bin ../data/glove.6B.300d.txt.word2vec newbatch.raw.tsv ../data/corrected.tsv ../data/gigaword4.5g.kenlm.bin lexsub specials
#mv scored.tsv newbatch.scored.tsv
#cut -f 1-44 newbatch.scored.tsv | perl -pe 's/\t/ /g' | grep -v "glove_src_para_sim" > newbatch.feats.tsv
#cut -f 52 newbatch.tsv | egrep -v "^annot" > newbatch.anno.tsv
#paste newbatch.anno.tsv newbatch.feats.tsv > newbatch.ready.tsv
#cat newbatch.ready.tsv | grep -P -v "^\t.*" > sanity.feats

#cat lexsub.ready.tsv newbatch.ready.tsv multiple.ready.tsv | grep -P -v "^\t.*" > sanity.feats

./score.sanity.py ../data/GoogleNews-vectors-negative300.bin ../data/glove.6B.300d.txt.word2vec combined.tsv ../data/corrected.tsv ../data/gigaword4.5g.kenlm.bin lexsub specials
mv scored.tsv combined.scored.tsv
cut -f 1-44 combined.scored.tsv | perl -pe 's/\t/ /g' | grep -v "glove_src_para_sim" > combined.feats.tsv
cut -f 52 combined.scored.tsv | egrep -v "^annot" > combined.anno.tsv
paste combined.anno.tsv combined.feats.tsv > combined.ready.tsv
cat combined.ready.tsv | grep -P -v "^\t.*" > sanity.feats

python score_noclass.py > sanity_output.tsv