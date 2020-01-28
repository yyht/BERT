from collections import Counter
from collections import defaultdict
import bisect
import numpy as np

def sample_index(ps_cumsum):
	return bisect.bisect(ps_cumsum, np.random.random() * ps_cumsum[-1])

def nram_counts(tokens, ngm_count_dict, ngram_n):
	for i in range(len(tokens)+1):
		for j in range(i):
			if i-j == ngram_n:
				ngram = tuple(tokens[j:i])
				if ngram in ngm_count_dict:
					ngm_count_dict[ngram] += 1
				else:
					ngm_count_dict[ngram] = 1
	return ngm_count_dict

def ngram_fdist(ngm_count_dict):
	total_unique_ngram = len(list(ngm_count_dict.keys()))
	total_ngram = 0
	for key in ngm_count_dict:
		total_ngram += ngm_count_dict[key]
	ngam_fdist = {}
	for key in ngm_count_dict:
		ngam_fdist[key] = ngm_count_dict[key] / total_ngram
	return ngam_fdist

def build_continuations(ngm_count_dict):
	total = defaultdict(int)
	distinct = defaultdict(int)
	for key in ngm_count_dict:
		context = key[:-1] # for bigram, just get the first token
		total[context] += ngm_count_dict[key]
		distinct[context] += 1
	return {"total": total, "distinct": distinct}

def estimate_modkn_discounts(ngm_count_dict):
	# Get counts
	counts = ngm_count_dict
	N1 = float(len([k for k in counts if counts[k] == 1]))
	N2 = float(len([k for k in counts if counts[k] == 2]))
	N3 = float(len([k for k in counts if counts[k] == 3]))
	N4 = float(len([k for k in counts if counts[k] == 4]))
	N3p = float(len([k for k in counts if counts[k] >= 3]))

	# Estimate discounting parameters
	Y = N1 / (N1 + 2 * N2)
	D1 = 1 - 2 * Y * (N2 / N1)
	D2 = 2 - 3 * Y * (N3 / N2)
	D3p = 3 - 4 * Y * (N4 / N3)

	# FIXME(zxie) Assumes bigrams for now
	# Also compute N1/N2/N3p lookups (context -> n-grams with count 1/2/3+)
	N1_lookup = Counter()
	N2_lookup = Counter()
	N3p_lookup = Counter()
	for bg in counts:
		if counts[bg] == 1:
			N1_lookup[bg[0]] += 1
		elif counts[bg] == 2:
			N2_lookup[bg[0]] += 1
		else:
			N3p_lookup[bg[0]] += 1

	return D1, D2, D3p, N1_lookup, N2_lookup, N3p_lookup

def ngram_hist(tokens, bg_hist_sets, ngram=2):
	for k in range(ngram-1, len(tokens)):
		bg_hist_sets[tokens[k]].add(tokens[k - ngram + 1])
	return bg_hist_sets

def nlm_distribution(token, continuations, gamma, 
											ngram_scheme,
											D1, D2, D3p, N1_lookup, N2_lookup, N3p_lookup, 
											hist_freqs_cumsum, hist_freq_cumsum_token):

	total, distinct = continuations["total"][tuple(token)],\
										continuations["distinct"][tuple(token)]
	if total == 0:
		total = 1
	if distinct == 0:
		distinct = 1
	if ngram_scheme != "mbgkn":
		p = (gamma / total) * distinct
	else:
		p = gamma * (D1 * N1_lookup[token[0]] +
								 D2 * N2_lookup[token[0]] +
								 D3p * N3p_lookup[token[0]]) / float(total)
	if p > 1:
		p = 1

	print(p, '==noisy probability==')

	draw = np.random.binomial(1, p)
	if draw:
		if np.random.random() < 0.8:
			masked_token = "[MASK]"
		else:
			masked_token = hist_freq_cumsum_token[sample_index(hist_freqs_cumsum)]
	else:
		masked_token = token
	return masked_token

	



