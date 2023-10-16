# %%
import argparse
import os.path
import pickle
import unicodedata

import torch
from tqdm import tqdm

import NER_medNLP as ner
import utils
from EntityNormalizer import EntityNormalizer, EntityDictionary, DefaultDiseaseDict, DefaultDrugDict

device = torch.device("mps" if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# %% global変数として使う
dict_key = {}


# %%
def to_xml(data, id_to_tags):
    with open("key_attr.pkl", "rb") as tf:
        key_attr = pickle.load(tf)

    text = data['text']
    count = 0
    for i, entities in enumerate(data['entities_predicted']):
        if entities == "":
            return
        span = entities['span']
        try:
            type_id = id_to_tags[entities['type_id']].split('_')
        except:
            print("out of rage type_id", entities)
            continue
        tag = type_id[0]

        if not type_id[1] == "":
            attr = ' ' + value_to_key(type_id[1], key_attr) + '=' + '"' + type_id[1] + '"'
        else:
            attr = ""

        if 'norm' in entities:
            attr = attr + ' norm="' + str(entities['norm']) + '"'

        add_tag = "<" + str(tag) + str(attr) + ">"
        text = text[:span[0] + count] + add_tag + text[span[0] + count:]
        count += len(add_tag)

        add_tag = "</" + str(tag) + ">"
        text = text[:span[1] + count] + add_tag + text[span[1] + count:]
        count += len(add_tag)
    return text


def predict_entities(modelpath, sentences_list, len_num_entity_type):
    model = ner.BertForTokenClassification_pl.from_pretrained_bin(model_path=modelpath, num_labels=2 * len_num_entity_type + 1)
    bert_tc = model.bert_tc.to(device)

    tokenizer = ner.NER_tokenizer_BIO.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking',
        num_entity_type=len_num_entity_type  # Entityの数を変え忘れないように！
    )

    # entities_list = [] # 正解の固有表現を追加していく
    entities_predicted_list = []  # 抽出された固有表現を追加していく

    text_entities_set = []
    for dataset in sentences_list:
        text_entities = []
        for sample in tqdm(dataset, desc='Predict'):
            text = sample
            encoding, spans = tokenizer.encode_plus_untagged(
                text, return_tensors='pt'
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                output = bert_tc(**encoding)
                scores = output.logits
                scores = scores[0].cpu().numpy().tolist()

            # 分類スコアを固有表現に変換する
            entities_predicted = tokenizer.convert_bert_output_to_entities(
                text, scores, spans
            )

            # entities_list.append(sample['entities'])
            entities_predicted_list.append(entities_predicted)
            text_entities.append({'text': text, 'entities_predicted': entities_predicted})
        text_entities_set.append(text_entities)
    return text_entities_set


def combine_sentences(text_entities_set, id_to_tags, insert: str):
    documents = []
    for text_entities in tqdm(text_entities_set):
        document = []
        for t in text_entities:
            document.append(to_xml(t, id_to_tags))
        documents.append('\n'.join(document))
    return documents


def value_to_key(value, key_attr):  # attributeから属性名を取得
    global dict_key
    if dict_key.get(value) != None:
        return dict_key[value]
    for k in key_attr.keys():
        for v in key_attr[k]:
            if value == v:
                dict_key[v] = k
                return k


# %%
def normalize_entities(text_entities_set, id_to_tags, disease_dict=None, disease_candidate_col=None, disease_normalization_col=None, disease_matching_threshold=None, drug_dict=None,
                       drug_candidate_col=None, drug_normalization_col=None, drug_matching_threshold=None):
    if disease_dict:
        disease_dict = EntityDictionary(disease_dict, disease_candidate_col, disease_normalization_col)
    else:
        disease_dict = DefaultDiseaseDict()
    disease_normalizer = EntityNormalizer(disease_dict, matching_threshold=disease_matching_threshold)

    if drug_dict:
        drug_dict = EntityDictionary(drug_dict, drug_candidate_col, drug_normalization_col)
    else:
        drug_dict = DefaultDrugDict()
    drug_normalizer = EntityNormalizer(drug_dict, matching_threshold=drug_matching_threshold)

    for entry in text_entities_set:
        for text_entities in entry:
            entities = text_entities['entities_predicted']
            for entity in entities:
                tag = id_to_tags[entity['type_id']].split('_')[0]

                normalizer = drug_normalizer if tag == 'm-key' \
                    else disease_normalizer if tag == 'd' \
                    else None

                if normalizer is None:
                    continue

                normalization, score = normalizer.normalize(entity['name'])
                entity['norm'] = str(normalization)


def run(model, input, output=None, normalize=False, **kwargs):
    with open("id_to_tags.pkl", "rb") as tf:
        id_to_tags = pickle.load(tf)

    if (os.path.isdir(input)):
        files = [f for f in os.listdir(input) if os.path.isfile(os.path.join(input, f))]
    else:
        files = [input]

    for file in tqdm(files, desc="Input file"):
        with open(file) as f:
            articles_raw = f.read()

        article_norm = unicodedata.normalize('NFKC', articles_raw)

        sentences_raw = utils.split_sentences(articles_raw)
        sentences_norm = utils.split_sentences(article_norm)

        text_entities_set = predict_entities(model, [sentences_norm], len(id_to_tags))

        for i, texts_ent in enumerate(text_entities_set[0]):
            texts_ent['text'] = sentences_raw[i]

        if normalize:
            normalize_entities(text_entities_set, id_to_tags, **kwargs)

        documents = combine_sentences(text_entities_set, id_to_tags, '\n')

        print(documents[0])

        if output:
            with open(file.replace(input, output), 'w') as f:
                f.write(documents[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict entities from text')
    parser.add_argument('-m', '--model', type=str, default='pytorch_model.bin', help='Path to model checkpoint')
    parser.add_argument('-i', '--input', type=str, default='text.txt', help='Path to text file or directory')
    parser.add_argument('-o', '--output', type=str, default=None, help='Path to output file or directory')
    parser.add_argument('-n', '--normalize', action=argparse.BooleanOptionalAction, help='Enable entity normalization', default=False)

    # Dictionary override arguments
    parser.add_argument("--drug-dict", help="File path for overriding the default drug dictionary")
    parser.add_argument("--drug-candidate-col", type=int, help="Column name for drug candidates in the CSV file (required if --drug-dict is specified)")
    parser.add_argument("--drug-normalization-col", type=int, help="Column name for drug normalization in the CSV file (required if --drug-dict is specified")
    parser.add_argument('--disease-matching-threshold', type=int, default=50, help='Matching threshold for disease dictionary')

    parser.add_argument("--disease-dict", help="File path for overriding the default disease dictionary")
    parser.add_argument("--disease-candidate-col", type=int, help="Column name for disease candidates in the CSV file (required if --disease-dict is specified)")
    parser.add_argument("--disease-normalization-col", type=int, help="Column name for disease normalization in the CSV file (required if --disease-dict is specified)")
    parser.add_argument('--drug-matching-threshold', type=int, default=50, help='Matching threshold for drug dictionary')
    args = parser.parse_args()

    argument_dict = vars(args)
    run(**argument_dict)
