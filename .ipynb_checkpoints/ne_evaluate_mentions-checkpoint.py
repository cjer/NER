from collections import defaultdict
from itertools import islice
import pandas as pd

def evaluate_mentions(true_ments, pred_ments, examples=5, verbose=True):
    t, p = set(true_ments), set(pred_ments)
    correct = p.intersection(t)

    prec = len(correct) / len(t)
    recall = len(correct) / len(p)
    f1 = 2*prec*recall/(prec+recall)
    if verbose:
        print(len(t), 'mentions,', len(p), 'found,', len(correct), 'correct.')
        print('Precision:', round(prec, 2))
        print('Recall:   ', round(recall, 2))
        print('F1:       ', round(f1, 2))
        print('FP ex.:', [e[1] for e in list(p-t)[:examples]])
        print('FN ex.:', [e[1] for e in list(t-p)[:examples]])
    return prec, recall, f1
        
        
def sent_to_mentions_dict(sent, sent_id, truncate=80):
    mentions = defaultdict(lambda: 0)
    current_mention= None
    current_cat = None
    if truncate is not None:
        it = islice(sent, truncate)
    else:
        it = sent
        
    for tok, bio, cat in it:
        if bio=='S':
            mentions[(sent_id, tok, cat)]+=1
        if bio=='B':
            current_mention = [tok]
            current_cat = cat
        if bio=='I' and current_mention is not None:
            current_mention.append(tok)
        if bio=='E' and current_mention is not None:
            current_mention.append(tok)
            mentions[(sent_id, ' '.join(current_mention), current_cat)]+=1
        if bio=='O':
            current_mention = None
            current_cat = None
    return mentions


def get_ment_set(ments):
    ment_set = []
    for ment in ments:
        for k, val in ment.items():
            for i in range(val):
                ment_set.append((k[0], k[1], k[2], i+1))
    return ment_set

def get_sents_fixed(sents):
    sf = []
    for sent in sents:
        new_sent = []
        for tok, biose in sent:
            tag = biose.split('-')
            biose = tag[0]
            if len(tag)>1:
                cat = tag[1]
            else:
                cat = '_'
            new_sent.append((tok, biose, cat))
        sf.append(new_sent)
    sf = list(zip(list(sents.index), sf))
    return sf

def sents_to_mentions(sents, truncate=80):
    sents_fixed = get_sents_fixed(sents)
    ments = [sent_to_mentions_dict(sent, sent_id, truncate) for sent_id, sent in sents_fixed]
    ment_set = get_ment_set(ments)
    return ment_set


def get_sents_with_pred_tags(splits, preds, truncate=80):
    sents_preds = []
    for split, pred in zip(splits, preds):
        spl_preds = []
        test_sents = split[3]
        
        i=0
        for sent in test_sents:
            new_sent = []
            for tok, bio, cat in islice(sent, truncate):
                pred_tag = pred[i].split('-')
                pred_bio = pred_tag[0]
                if len(pred_tag)>1:
                    pred_cat = pred_tag[1]
                else:
                    pred_cat = '_'
                new_sent.append((tok, pred_bio, pred_cat))
                i+=1
            spl_preds.append(new_sent)
        spl_preds = pd.Series(spl_preds, index=test_sents.index)
        sents_preds.append(spl_preds)
    return sents_preds
