# General book keeping

## COSI plan:
A/B test---how now?

## Needed:
- [ ] COSI data

## Implementation
- [ ] many bug fixes. Primary:
	- [ ] NAN problem
- [ ] Sampling schema:
	- [x] Currently sampling UP TO 20 per infreq label
	- [ ] Percent?
	- [ ] Scaling

## Next up:
- [ ] Run VP data with all (?) metrics through RNN?
- [ ] Generate paraphrases from COSI data---run through RNN

## Ideas:
- [ ] KNN over sentence embeddings!
- [ ] All 4 concated!
- [ ] Test with new transformer classifier?

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
Sampled up to 20 on infrequent labels
```
------------------------------------------------------------------------------------
| dev average | test loss (pure) 1.3314 | Acc   0.8351
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
| test average | test loss (pure) 1.3972 | Acc   0.8263
------------------------------------------------------------------------------------
```

| Fold             |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |
| :--------------: |  :-:  |  :-:  |  :-:  |  :-:  |  :-:  |  :-:  |  :-:  |  :-:  |  :-:  |  :-:  |
| w/o sampling Acc | 81.11 | 82.24 | 80.09 | 81.11 | 79.94 | 80.09 | 83.75 | 79.18 | 83.87 | 80.23 |
| w/ sampling Acc  | 83.45 | 81.99 | 80.97 | 83.46 | 81.84 | 80.96 | 84.77 | 81.96 | 84.75 | 82.87 |

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
