
Week10
Workshop on Nov. 24:

In this workshop, we produce the classification baseline from the Adversarial Debiasing
Paper: Zhang, Lemoine & Mitchell (2018), Mitigating Unwanted Biases with Adversarial Learning.

For this workshop, the computing time is small. What takes time is understanding the results. Take your time, and brainstorm with your teammates before you write each answer. Many questions don't have a single number as the answer: they ask why something happens, or which method is better and in what situation. A good answer explains the result, not just reports it.

First, trains three models on UCI Adult and compares them (3 runs)

1. Baseline (no debiasing)
2. Debiased for demographic parity (adversary sees \hat{Y} only)
3. Debiased for equalized odds (adversary sees \hat{Y} and true Y)

How to run in Google Colab

1. !python main.py
2. Show the figures in a cell:
   from IPython.display import Image, display
   display(Image("comparison.png")); display(Image("curves.png"))

Questions:

While training, watch L_A (the adversary's loss).
Q1) How the loss is changing in parity mode it and in equalized-odds mode?
Q2) For the baseline and the parity model, plot histograms of the predicted probability sigmoid(P(x)) separately for men and women (two overlapping histograms per model). Save them as scores.png. How does debiasing change the two distributions? (2 runs)
Q3) Parity mode nearly closes the DP gap, but what happens to its TPR gap? Why?
  (Hint: base rates differ: P(y=1|M) ~ 0.31 vs P(y=1|F) ~ 0.11.)
Q4) When base rates differ, can a non-trivial classifier satisfy demographic
  parity AND equalized odds at the same time?
Q5) Why is the baseline still biased even though 'sex`was removed from the inputs? Find the features most correlated with`sex`. Q6) Give the adversary a hidden layer. Does this make debiasing more effective, or less stable? (2 runs) Q7) The paper grows the adversary weight over time (~ sqrt(t)). Implement this and compare the results with the baseline. (1 run) Q8) Let's use race instead of sex. In data.py, make z = 1 for "White" and 0 for everyone else, using the `race`column. Remove`race`from the features, but keep`sex` as a normal feature this time. Rerun main.py. Is the baseline more or less biased for race than for sex. (3 runs)
Q9) Equal opportunity. Add a third adversary mode "eq_opp" that only cares about the TPR gap: compute the adversary's loss on the y = 1 examples only (mask the batch). Compare its TPR and FPR gaps with "eq_odds".(1 run)
Q10) Gradient reversal. Many papers (Ganin et al., 2016) use a "gradient reversal layer" instead of the projection: a layer that is the identity in the forward pass and multiplies the gradient by -alpha in the backward pass. Implement it with torch.autograd.Function, put it between the predictor and the adversary, and train with ONE ordinary loss.backward(). Compare with the baseline. (1 run)
Q11) Instead of giving the adversary the predictor's output, give it the predictor's last hidden layer (64 numbers). Does the parity gap still close? Is training more or less stable? (1 run)
Q12) Take the Baseline model and choose a separate threshold for men and women so the DP gap on the validation set is close to 0. Compare accuracy and gaps with the adversarial parity model. Which is simpler? Which is better? (2 runs)
