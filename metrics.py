from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf

def evaluate_model(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'])
    rouge_scores = [scorer.score(r, p) for r, p in zip(references, predictions)]
    avg_rouge = {
        key: sum(score[key].fmeasure for score in rouge_scores)/len(predictions)
        for key in rouge_scores[0]
    }

    bleu = corpus_bleu([[ref.split()] for ref in references], [pred.split() for pred in predictions])
    chrf = corpus_chrf(references, predictions)

    return {'ROUGE': avg_rouge, 'BLEU': bleu, 'ChrF': chrf}
