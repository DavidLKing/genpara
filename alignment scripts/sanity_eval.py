import pdb
import pandas

sanity_file = open('sanity.tsv', 'r').readlines()
header = sanity_file.pop(0)
sanity_file = [x.strip().split('\t') for x in sanity_file]
sanity = pandas.DataFrame(sanity_file, columns=header.split('\t'))

def get_values(annotations, to_eval):
    prec_num = 0
    prec_denom = 0
    rec_num = 0
    rec_denom = 0
    f1 = lambda p, r: 2 * ((p * r) / (p + r))
    for i in range(to_eval):
        if annotations[i] in ['o', 'p']:
            prec_num += 1
            rec_num += 1
        prec_denom += 1
    for annot in annotations:
        if annot in ['o', 'p']:
            rec_denom += 1
    try:
        prec = prec_num / prec_denom
    except:
        prec = 0
    try:
        rec = rec_num / rec_denom
    except:
        rec = 0
    try:
        harm_mean = f1(prec, rec)
    except:
        harm_mean = 0
    # print('\tprecision:', prec)
    # print('\trecall:', rec)
    # print('\tf1:', harm_mean)
    return prec, rec, harm_mean


def prec_rec(annotations):
    break_down = []
    for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        # print('Eval at top', str(percent * 100) + '%:')
        prec, rec, harm_mean = get_values(annotations, int(len(annotations) * percent))
        break_down.append([percent, prec, rec, harm_mean])
    return break_down

eval_list = []
for metric in ['glove_src_para_sim', 'glove_src_para_dist', 'glove_src_para_david',
       'glove_src_orig_sim', 'glove_src_orig_dist', 'glove_src_orig_david',
       'glove_orig_para_sim', 'glove_orig_para_dist', 'glove_orig_para_david',
       'glove_align_para_sim', 'glove_align_para_dist',
       'glove_align_para_david', 'w2v_src_para_sim', 'w2v_src_para_dist',
       'w2v_src_para_david', 'w2v_src_orig_sim', 'w2v_src_orig_dist',
       'w2v_src_orig_david', 'w2v_orig_para_sim', 'w2v_orig_para_dist',
       'w2v_orig_para_david', 'w2v_align_para_sim', 'w2v_align_para_dist',
       'w2v_align_para_david', 'elmo_src_para_sim', 'elmo_src_para_dist',
       'elmo_src_para_david', 'elmo_src_orig_sim', 'elmo_src_orig_dist',
       'elmo_src_orig_david', 'elmo_orig_para_sim', 'elmo_orig_para_dist',
       'elmo_orig_para_david', 'elmo_align_para_sim', 'elmo_align_para_dist',
       'elmo_align_para_david']:
    # print("Metric:", metric)
    if metric[-3] == 'sim':
        sanity = sanity.sort_values(by=[metric], ascending=False)
    else:
        sanity = sanity.sort_values(by=[metric], ascending=True)
    annotations = sanity['okay/perf/bad'].values.tolist()
    break_down = prec_rec(annotations)
    break_down = [[metric] + x for x in break_down]
    eval_list.append(break_down)
    # pdb.set_trace()

header = ['metric', 'percent kept', 'prec', 'rec', 'f1']
print('\t'.join(header))
for metric in eval_list:
    for percentage in metric:
        print('\t'.join([str(x) for x in percentage]))
# pdb.set_trace()