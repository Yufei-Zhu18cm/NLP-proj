import sacrebleu

def corpus_bleu(hyps, refs):
    # refs: list[str], hyps: list[str]
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score
