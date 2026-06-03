Permutation convention:
pi[i] = a means vertex i in graph A corresponds to vertex a in graph B.

Given pi, the aligned version of B is:
B_aligned = B[np.ix_(pi, pi)]

The default score is:
S(pi) = sum_{i<j} A[i,j] * B[pi[i], pi[j]]

The MCMC target is:
p_beta(pi | A,B) proportional to exp(beta * S(pi)).

Accuracy is:
mean(pi_hat == pi_true)

Important:
The inference algorithm should use only A and B.
pi_true and B_clean are only for debugging/evaluation.
