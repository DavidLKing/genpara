import pdb
import sys
import pandas as pd

datas = pd.read_csv(sys.argv[1], sep='\t')

# top_X_2 = datas[datas["log_0_2"]>=0.9]
# top_X_4 = datas[datas["log_0_4"]>=0.9]

tenth = len(datas.para.tolist()) // 10

top_X_2 = datas.sort_values(by=['log_1_2'], ascending=0).head(tenth)
top_X_4 = datas.sort_values(by=['log_1_4'], ascending=0).head(tenth)

top_X_2.to_csv('../../../StructSelfAttnSentEmb/data/para_scores_2_top.tsv', sep='\t')
top_X_4.to_csv('../../../StructSelfAttnSentEmb/data/para_scores_4_top.tsv', sep='\t')

print("Done")
