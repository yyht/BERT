# bert
modification of official bert for downstream task

# Support OQMRC, LCQMC, knowledge distillation, adversarial disturbation and bert+esim for multi-choice, classification and semantic match

for OQMRC, we can get 0.787% on dev set
for LCQMC, we can get 0.864 on test set
knowledge distillation supports self-distillation

# support task pretrain+fintuning
for a downstream task, we add masked lm as a auxiliary loss which can be seen as denoising and similar to word dropout to achieve robust performance.

