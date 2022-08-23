
# %%
from tqdm import tqdm
import unicodedata
import re
import pickle
import torch
import NER_medNLP as ner
from bs4 import BeautifulSoup


# import from_XML_to_json as XtC
# import itertools
# import random
# import json
# from torch.utils.data import DataLoader
# from transformers import BertJapaneseTokenizer, BertForTokenClassification
# import pytorch_lightning as pl
# import pandas as pd
# import numpy as np
# import codecs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%% global変数として使う
dict_key = {}

#%%
def to_xml(data):
    with open("key_attr.pkl", "rb") as tf:
        key_attr = pickle.load(tf)
    
    text = data['text']
    count = 0
    for i, entities in enumerate(data['entities_predicted']):
        if entities == "":
            return   
        span = entities['span']
        type_id = id_to_tags[entities['type_id']].split('_')
        tag = type_id[0]
        
        if not type_id[1] == "":
            attr = ' ' + value_to_key(type_id[1], key_attr) +  '=' + '"' + type_id[1] + '"'
        else:
            attr = ""
        
        add_tag = "<" + str(tag) + str(attr) + ">"
        text = text[:span[0]+count] + add_tag + text[span[0]+count:]
        count += len(add_tag)

        add_tag = "</" + str(tag) + ">"
        text = text[:span[1]+count] + add_tag + text[span[1]+count:]
        count += len(add_tag)
    return text


def predict_entities(modelpath, sentences_list, len_num_entity_type):
    # model = ner.BertForTokenClassification_pl.load_from_checkpoint(
    #     checkpoint_path = modelpath + ".ckpt"
    # ) 
    # bert_tc = model.bert_tc.cuda()
    
    model = ner.BertForTokenClassification_pl(modelpath, num_labels=81, lr=1e-5) 
    bert_tc = model.bert_tc.to(device)

    MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = ner.NER_tokenizer_BIO.from_pretrained(
        MODEL_NAME,
        num_entity_type =  len_num_entity_type#Entityの数を変え忘れないように！
    )

    #entities_list = [] # 正解の固有表現を追加していく
    entities_predicted_list = [] # 抽出された固有表現を追加していく

    text_entities_set = []
    for dataset in sentences_list:
        text_entities = []
        for sample in tqdm(dataset):
            text = sample
            encoding, spans = tokenizer.encode_plus_untagged(
                text, return_tensors='pt'
            )
            encoding = { k: v.to(device) for k, v in encoding.items() } 
            
            with torch.no_grad():
                output = bert_tc(**encoding)
                scores = output.logits
                scores = scores[0].cpu().numpy().tolist()
                
            # 分類スコアを固有表現に変換する
            entities_predicted = tokenizer.convert_bert_output_to_entities(
                text, scores, spans
            )

            #entities_list.append(sample['entities'])
            entities_predicted_list.append(entities_predicted)
            text_entities.append({'text': text, 'entities_predicted': entities_predicted})
        text_entities_set.append(text_entities)
    return text_entities_set

def combine_sentences(text_entities_set, insert: str):
    documents = []
    for text_entities in tqdm(text_entities_set):
        document = []
        for t in text_entities:
            document.append(to_xml(t))
        documents.append('\n'.join(document))
    return documents

def value_to_key(value, key_attr):#attributeから属性名を取得
    global dict_key
    if dict_key.get(value) != None:
        return dict_key[value]
    for k in key_attr.keys():
        for v in key_attr[k]:
            if value == v:
                dict_key[v]=k
                return k

# %%
if __name__ == '__main__':
    with open("id_to_tags.pkl", "rb") as tf:
        id_to_tags = pickle.load(tf)
    with open("key_attr.pkl", "rb") as tf:
        key_attr = pickle.load(tf)
    with open('text.txt') as f:
        articles_raw = f.read()
        

    article_norm = unicodedata.normalize('NFKC', articles_raw)

    sentences_raw = [s for s in re.split(r'\n', articles_raw) if s != '']
    sentences_norm = [s for s in re.split(r'\n', article_norm) if s != '']

    text_entities_set = predict_entities("sociocom/RealMedNLP_CR_JA", [sentences_norm], len(id_to_tags))


    for i, texts_ent in enumerate(text_entities_set[0]):
        texts_ent['text'] = sentences_raw[i]


    documents = combine_sentences(text_entities_set, '\n')

    print(documents[0])
