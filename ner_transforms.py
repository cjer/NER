import numpy as np
import pandas as pd


def get_mention_id_sents(ner):
    from collections import defaultdict
    curr_sent = -1

    def zero():
        return 0
    tags_idx = defaultdict(zero)
    old_to_new = {}
    new_tags = []
    sents = []
    sent = None
    for (sent_id, morph_id), tags in ner.iteritems():
        if curr_sent!=sent_id:
            sents.append(sent)
            sent = []
            tags_idx = defaultdict(zero)
            old_to_new = {}
            curr_sent=sent_id

        ftags = []
        for tag in tags:
            spl = tag.split('[')
            typ = spl[0]
            if len(spl)==1:
                ftags.append((tag+str(tags_idx[typ])))
                tags_idx[typ]+=1
            else:
                old_idx = int(spl[1][:-1])
                if old_idx in old_to_new:
                    new_idx = old_to_new[old_idx]
                else:
                    new_idx = tags_idx[typ]
                    tags_idx[typ]+=1
                    old_to_new[old_idx] = new_idx
                ftags.append((typ+str(new_idx)))
        sent.append(ftags)
    sents.append(sent)
    sents = sents[1:]
    return sents


_flatten = lambda l: [item for sublist in l for item in sublist]


def get_layer_dicts(sents):
    layers = []
    for tags in sents:
        d = {}
        for i, tag in enumerate(tags):
            d['layer'+str(i)] = tag
        layers.append(d)
    return layers


def find_bad_layers(g, sent_id_col):
    layer_columns = [c for c in g.columns if c.startswith('layer')]
    ents = {}
    for l in layer_columns:
        for e in g[l]:
            if e is not None and e is not np.nan and not e.startswith('_'):
                if e in ents and l not in ents[e]:
                    ents[e].append(l)
                else:
                    ents[e] = [l]
    bad_sent=False
    for e, l in ents.items():
        if len(l)>1:
            print (g[sent_id_col].iat[0], e, l)
            bad_sent=True
    if bad_sent:
        print(g[sent_id_col].iat[0], 'bad sentence') 
        return True
    return False
    
    
def create_biose(cat, len):
    if len<1:
        raise ValueError
    if len==1:
        return ['S-'+cat]
    else:
        return ['B-'+cat]+['I-'+cat]*(len-2)+['E-'+cat]

    
def sent_layer_tags_to_biose(sent_layer_tags):
    ents = {}
    for i, tag in enumerate(sent_layer_tags.values):
        if tag is not None and tag is not np.nan:
            if tag in ents:
                ents[tag]['end']=i
            else:
                ents[tag] = {'start': i, 'end': i}

    n = sent_layer_tags.shape[0]
    i = -1            
    if len(ents)==0:
        return ['O']*(n-i-1)
    
    sent_df = pd.DataFrame([{'ent': ent, 'start': se['start'], 'end': se['end']} for ent, se in ents.items()])[['ent', 'start', 'end']].sort_values(['start'], ascending=True)

    biose = []
    for ent, start, end in sent_df.values:
        cat = ent[:3]
        biose.extend(['O']*(start-i-1))
        i = end
        biose.extend(create_biose(cat, end-start+1))
    biose.extend(['O']*(n-i-1))
    return biose

def get_layers_biose(sent):
    layer_columns = [c for c in sent.columns if c.startswith('layer')]
    layers = []
    for col in layer_columns:
        layer_tags = sent_layer_tags_to_biose(sent[col])
        layers.append(pd.Series(layer_tags, name='biose_'+col))
    
    return pd.concat(layers, axis=1)


def get_all_layers_biose(tsv3_tags, sent_id_col='sent_id', morpheme_id_col='id', delim='|', ner_tag_col='ner'):
    ner = tsv3_tags.set_index([sent_id_col, morpheme_id_col])[ner_tag_col].str.split('|')
    sents = get_mention_id_sents(ner)
    x = ner.reset_index()
    fsents = _flatten(sents)
    fsents = pd.DataFrame(get_layer_dicts(fsents))
    x = pd.concat([x, fsents], axis=1)
    if x.groupby([sent_id_col]).apply(find_bad_layers, sent_id_col).any(axis=0):
        raise ValueError
    x.loc[x.layer0.str.startswith('_'), 'layer0'] = None
    biose = x.groupby(sent_id_col).apply(get_layers_biose).reset_index().drop(['level_1', sent_id_col], axis=1)
    
    return pd.concat([x[[sent_id_col, morpheme_id_col]], biose], axis=1)