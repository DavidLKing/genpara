import pdb
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids


def dist(a, b):
    return np.linalg.norm(a - b)

def sim(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim

def joint(a, b):
    return (1 -(sim(a, b) + 1) / 2) * dist(a, b)

# elmo = Elmo('../data/elmo_2x4096_512_2048cnn_2xhighway_options.json', '../data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', 2)
elmo = Elmo('../data/bigelmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json', '../data/bigelmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5', 2)
elmo = elmo.cuda(device=0)

src = '<S> I have a bug in my stomach </S>'.split()
align = '<S> I have a stomach illness </S>'.split()
orig_bad = '<S> My code has a bug </S>'.split()
orig_good = "<S> I 've come down with a bug and I feel terrible </S>".split()
para_bad = "<S> My code has an illness </S>".split()
para_good = "<S> I 've come down with an illness and I feel terrible </S>".split()

src_other = '<S> I would like to ask about your prior medical history </S>'.split()
align_other = '<S> Can I ask you about your past medical history </S>'.split()
orig_bad_other = '<S> You should check the prior in your equation </S>'.split()
orig_good_other = "<S> Tell me about any prior problems </S>".split()
para_bad_other = "<S> You should check the past in your equation </S>".split()
para_good_other = "<S> Tell me about any past problems </S>".split()

# stupid warmup
warmup = 100
for i in range(warmup):
    if i % 10 == 0:
        print("on", i, 'of', warmup)
    for sent in [src, align, orig_bad, orig_good, para_bad, para_good,
            src_other, align_other, orig_bad_other, orig_good_other, para_bad_other, para_good_other]:
        # elmo(batch_to_ids([sent]))
        elmo(batch_to_ids([sent]).cuda(device=0))


for sent, sent_type in zip(
        [src, align, orig_bad, orig_good, para_bad, para_good],
        ['src', 'align', 'orig_bad', 'orig_good', 'para_bad', 'para_good']
        ):
    print(sent_type + ":", sent)

print()


src_elm = elmo(batch_to_ids([src]).cuda(device=0))
align_elm = elmo(batch_to_ids([align]).cuda(device=0))
orig_bad_elm = elmo(batch_to_ids([orig_bad]).cuda(device=0))
orig_good_elm = elmo(batch_to_ids([orig_good]).cuda(device=0))
para_bad_elm = elmo(batch_to_ids([para_bad]).cuda(device=0))
para_good_elm = elmo(batch_to_ids([para_good]).cuda(device=0))

# src_elm = elmo(batch_to_ids([src]))
# align_elm = elmo(batch_to_ids([align]))
# orig_bad_elm = elmo(batch_to_ids([orig_bad]))
# orig_good_elm = elmo(batch_to_ids([orig_good]))
# para_bad_elm = elmo(batch_to_ids([para_bad]))
# para_good_elm = elmo(batch_to_ids([para_good]))

src_emb = src_elm['elmo_representations'][-1][0][4].detach().cpu().numpy()
align_emb = align_elm['elmo_representations'][-1][0][5].detach().cpu().numpy()
orig_bad_emb = orig_bad_elm['elmo_representations'][-1][0][5].detach().cpu().numpy()
orig_good_emb = orig_good_elm['elmo_representations'][-1][0][7].detach().cpu().numpy()
para_bad_emb = para_bad_elm['elmo_representations'][-1][0][5].detach().cpu().numpy()
para_good_emb = para_good_elm['elmo_representations'][-1][0][7].detach().cpu().numpy()

# src_emb = src_elm['elmo_representations'][-1][0][4].detach().numpy()
# align_emb = align_elm['elmo_representations'][-1][0][5].detach().numpy()
# orig_bad_emb = orig_bad_elm['elmo_representations'][-1][0][5].detach().numpy()
# orig_good_emb = orig_good_elm['elmo_representations'][-1][0][7].detach().numpy()
# para_bad_emb = para_bad_elm['elmo_representations'][-1][0][5].detach().numpy()
# para_good_emb = para_good_elm['elmo_representations'][-1][0][7].detach().numpy()


src_word = src[4]
align_word = align[5]
orig_bad_word = orig_bad[5]
orig_good_word = orig_good[7]
para_bad_word = para_bad[5]
para_good_word = para_good[7]


print('\t'.join(['src_word', src_word, 'align_word', align_word, 'cos_sim', str(sim(src_emb, align_emb)), 'dist', str(dist(src_emb, align_emb)), 'joint', str(joint(src_emb, align_emb))]))

print()

print('\t'.join(['src_word', src_word, 'orig_bad_word', orig_bad_word, 'cos_sim', str(sim(src_emb, orig_bad_emb)), 'dist', str(dist(src_emb, orig_bad_emb)), 'joint', str(joint(src_emb, orig_bad_emb))]))
print('\t'.join(['src_word', src_word, 'orig_good_word', orig_good_word, 'cos_sim', str(sim(src_emb, orig_good_emb)), 'dist', str(dist(src_emb, orig_good_emb)), 'joint', str(joint(src_emb, orig_good_emb))]))

print()

print('\t'.join(['src_word', src_word, 'para_bad_word', para_bad_word, 'cos_sim', str(sim(src_emb, para_bad_emb)), 'dist', str(dist(src_emb, para_bad_emb)), 'joint', str(joint(src_emb, para_bad_emb))]))
print('\t'.join(['src_word', src_word, 'para_good_word', para_good_word, 'cos_sim', str(sim(src_emb, para_good_emb)), 'dist', str(dist(src_emb, para_good_emb)), 'joint', str(joint(src_emb, para_good_emb))]))

print()

print('\t'.join(['align_word', align_word, 'orig_bad_word', orig_bad_word, 'cos_sim', str(sim(align_emb, orig_bad_emb)), 'dist', str(dist(align_emb, orig_bad_emb)), 'joint', str(joint(align_emb, orig_bad_emb))]))
print('\t'.join(['align_word', align_word, 'orig_good_word', orig_good_word, 'cos_sim', str(sim(align_emb, orig_good_emb)), 'dist', str(dist(align_emb, orig_good_emb)), 'joint', str(joint(align_emb, orig_good_emb))]))

print()

print('\t'.join(['align_word', align_word, 'para_bad_word', para_bad_word, 'cos_sim', str(sim(align_emb, para_bad_emb)), 'dist', str(dist(align_emb, para_bad_emb)), 'joint', str(joint(align_emb, para_bad_emb))]))
print('\t'.join(['align_word', align_word, 'para_good_word', para_good_word, 'cos_sim', str(sim(align_emb, para_good_emb)), 'dist', str(dist(align_emb, para_good_emb)), 'joint', str(joint(align_emb, para_good_emb))]))


print()


for sent, sent_type in zip(
        [src_other, align_other, orig_bad_other, orig_good_other, para_bad_other, para_good_other],
        ['src', 'align', 'orig_bad', 'orig_good', 'para_bad', 'para_good']
        ):
    print(sent_type + ":", sent)

print()

src_elm_other = elmo(batch_to_ids([src_other]).cuda(device=0))
align_elm_other = elmo(batch_to_ids([align_other]).cuda(device=0))
orig_bad_elm_other = elmo(batch_to_ids([orig_bad_other]).cuda(device=0))
orig_good_elm_other = elmo(batch_to_ids([orig_good_other]).cuda(device=0))
para_bad_elm_other = elmo(batch_to_ids([para_bad_other]).cuda(device=0))
para_good_elm_other = elmo(batch_to_ids([para_good_other]).cuda(device=0))

# src_elm = elmo(batch_to_ids([src]))
# align_elm = elmo(batch_to_ids([align]))
# orig_bad_elm = elmo(batch_to_ids([orig_bad]))
# orig_good_elm = elmo(batch_to_ids([orig_good]))
# para_bad_elm = elmo(batch_to_ids([para_bad]))
# para_good_elm = elmo(batch_to_ids([para_good]))

src_emb_other = src_elm_other['elmo_representations'][-1][0][8].detach().cpu().numpy()
align_emb_other = align_elm_other['elmo_representations'][-1][0][7].detach().cpu().numpy()
orig_bad_emb_other = orig_bad_elm_other['elmo_representations'][-1][0][5].detach().cpu().numpy()
orig_good_emb_other = orig_good_elm_other['elmo_representations'][-1][0][5].detach().cpu().numpy()
para_bad_emb_other = para_bad_elm_other['elmo_representations'][-1][0][5].detach().cpu().numpy()
para_good_emb_other = para_good_elm_other['elmo_representations'][-1][0][5].detach().cpu().numpy()

# src_emb = src_elm['elmo_representations'][-1][0][4].detach().numpy()
# align_emb = align_elm['elmo_representations'][-1][0][5].detach().numpy()
# orig_bad_emb = orig_bad_elm['elmo_representations'][-1][0][5].detach().numpy()
# orig_good_emb = orig_good_elm['elmo_representations'][-1][0][7].detach().numpy()
# para_bad_emb = para_bad_elm['elmo_representations'][-1][0][5].detach().numpy()
# para_good_emb = para_good_elm['elmo_representations'][-1][0][7].detach().numpy()

src_word_other = src_other[8]
align_word_other = align_other[7]
orig_bad_word_other = orig_bad_other[5]
orig_good_word_other = orig_good_other[5]
para_bad_word_other = para_bad_other[5]
para_good_word_other = para_good_other[5]


print('\t'.join(['src_word', src_word_other, 'align_word', align_word_other, 'cos_sim', str(sim(src_emb_other, align_emb_other)), 'dist', str(dist(src_emb_other, align_emb_other)), 'joint', str(joint(src_emb_other, align_emb_other))]))

print()

print('\t'.join(['src_word', src_word_other, 'orig_bad_word', orig_bad_word_other, 'cos_sim', str(sim(src_emb_other, orig_bad_emb_other)), 'dist', str(dist(src_emb_other, orig_bad_emb_other)), 'joint', str(joint(src_emb_other, orig_bad_emb_other))]))
print('\t'.join(['src_word', src_word_other, 'orig_good_word', orig_good_word_other, 'cos_sim', str(sim(src_emb_other, orig_good_emb_other)), 'dist', str(dist(src_emb_other, orig_good_emb_other)), 'joint', str(joint(src_emb_other, orig_good_emb_other))]))

print()

print('\t'.join(['src_word', src_word_other, 'para_bad_word', para_bad_word_other, 'cos_sim', str(sim(src_emb_other, para_bad_emb_other)), 'dist', str(dist(src_emb_other, para_bad_emb_other)), 'joint', str(joint(src_emb_other, para_bad_emb_other))]))
print('\t'.join(['src_word', src_word_other, 'para_good_word', para_good_word_other, 'cos_sim', str(sim(src_emb_other, para_good_emb_other)), 'dist', str(dist(src_emb_other, para_good_emb_other)), 'joint', str(joint(src_emb_other, para_good_emb_other))]))

print()

print('\t'.join(['align_word', align_word_other, 'orig_bad_word', orig_bad_word_other, 'cos_sim', str(sim(align_emb_other, orig_bad_emb_other)), 'dist', str(dist(align_emb_other, orig_bad_emb_other)), 'joint', str(joint(align_emb_other, orig_bad_emb_other))]))
print('\t'.join(['align_word', align_word_other, 'orig_good_word', orig_good_word_other, 'cos_sim', str(sim(align_emb_other, orig_good_emb_other)), 'dist', str(dist(align_emb_other, orig_good_emb_other)), 'joint', str(joint(align_emb_other, orig_good_emb_other))]))

print()

print('\t'.join(['align_word', align_word_other, 'para_bad_word', para_bad_word_other, 'cos_sim', str(sim(align_emb_other, para_bad_emb_other)), 'dist', str(dist(align_emb_other, para_bad_emb_other)), 'joint', str(joint(align_emb_other, para_bad_emb_other))]))
print('\t'.join(['align_word', align_word_other, 'para_good_word', para_good_word_other, 'cos_sim', str(sim(align_emb_other, para_good_emb_other)), 'dist', str(dist(align_emb_other, para_good_emb_other)), 'joint', str(joint(align_emb_other, para_good_emb_other))]))

pdb.set_trace()
