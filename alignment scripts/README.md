# General book keeping


[Last meeting written notes:]](:/14706dcee1a247f3babf5518a2ee5cb4)
- Switchboard dialog tagging dataset
	- Vashal using it---loot at and send him your datasets
- Mike has preliminary/pilot questions

## COSI plan:
A/B test---how now?

## Needed:
- [ ] COSI data

## Implementation
- [ ] many bug fixes (scoring). Primary:
	- [ ] NAN problem
- [ ] Add logging interface (i.e. Tensorboard) for Adam-RNN(TM)

## Next up:
- [ ] **from mike** Pretrain on uniform dist, then fine tune on empiracal dist
- [ ] 1/x training
- [ ] Run VP data with all (?) metrics through RNN?
- [ ] Analysis on output
- [x] Look at more recent paraphrasing papers. What are people doing? What's a good baseline

## Ideas:
- [ ] Generate paraphrases from COSI data---run through RNN
- [ ] KNN over sentence embeddings!
- [ ] All 4 concated!
- [x] Test with new transformer classifier?


__________________
# Alignment scrips

## General notes:
150+ files --> 8
10k lines --> 1k

No sampling:
```
------------------------------------------------------------------------------------
| dev average | test loss (pure) 1.3233 | Acc   0.8244
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
| test average | test loss (pure) 1.4114 | Acc   0.8135
------------------------------------------------------------------------------------
```
Small 41 addition
```
------------------------------------------------------------------------------------
| dev average | test loss (pure) 1.3323 | Acc   0.8264
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
| test average | test loss (pure) 1.4154 | Acc   0.8163
------------------------------------------------------------------------------------
```
Added up to 20 on infrequent labels
```
------------------------------------------------------------------------------------
| dev average | test loss (pure) 1.3314 | Acc   0.8351
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
| test average | test loss (pure) 1.3972 | Acc   0.8263
------------------------------------------------------------------------------------
```

0.1 sampling:
```
------------------------------------------------------------------------------------
| dev average | test loss (pure) 1.1734 | Acc   0.8371
------------------------------------------------------------------------------------
```
0.2 sampling:
```
------------------------------------------------------------------------------------
| dev average | test loss (pure) 1.1815 | Acc   0.8381
------------------------------------------------------------------------------------
```
0.3 sampling:
```
------------------------------------------------------------------------------------
| dev average | test loss (pure) 1.2179 | Acc   0.8392
------------------------------------------------------------------------------------
```
0.4 sampling:
```
------------------------------------------------------------------------------------
| dev average | test loss (pure) 1.1216 | Acc   0.8398
------------------------------------------------------------------------------------
```
0.5 sampling:
```
------------------------------------------------------------------------------------
| dev average | test loss (pure) 1.1892 | Acc   0.8396
------------------------------------------------------------------------------------
```
___
## Old todos:
- [x] Add sampling to Adam^tm RNN
- [x] Get scoring into new streamline
- [x] Finish cleaning genpara repo
- [x] What's the state with COSI and virtual stuff
- [x] Reach out to Laura, CC Mike and Eric and William
	- [x] What's the state of the pod and offerings
	- [x] We know where's some virtual thing
	- [x] Fall semester plan to deploy, Spring deploy
- [x] Sampling schema:
	- [x] Currently sampling UP TO 20 per infreq label
	- [x] **At 20% sampling, samply randomly from top 20%, not 20 percent** see Adam and Prashant's code
