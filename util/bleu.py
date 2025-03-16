import math
from collections import Counter

import numpy as np


def bleu_stats(hypothesis, reference):
    """
    compute BLEU stats for pairs of hypothesis and reference
    Args:
        hypothesis: list of tokens
        reference: list of tokens
    Returns:
        stats: list
    """
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats

def bleu(stats):
    """
    compute BLEU score from BLEU stats
    Args:
        stats: list
    Returns:
        bleu: float    
    """
    if (len(list(filter(lambda x: x == 0, stats))) > 0):
        return 0
    (c, r) = stats[:2]

    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    )

    bp = min(1 - float(r) / c, 0)

    return math.exp(bp + log_bleu_prec)

def get_bleu(hypotheses, reference):
    """
    get BLEU score for a list of hypotheses and references
    Args:
        hypotheses: list of list of tokens
        reference: list of list of tokens
    Returns:
        bleu: float
    """
    stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)
    

def idx_to_word(x, vocab):
    """
    convert index to word
    Args:
        x: list of indices
        vocab: torchtext.vocab.Vocab
    Returns:
        words: str
    """
    words = []
    for i in x:
        word = vocab.itos[i]
        if '<' in word:
            continue
        words.append(word)
    words = ' '.join(words)
    return words