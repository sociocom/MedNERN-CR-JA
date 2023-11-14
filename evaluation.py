# %%
import argparse
import json
import pickle

import torch
from tqdm import tqdm

import NER_medNLP as ner

device = torch.device("mps" if torch.backends.mps.is_available() else 'cuda:1' if torch.cuda.is_available() else 'cpu')

f_scores = []


# %%
# 8-19

def evaluate_model(entities_list, entities_predicted_list, type_id=None):
    """
    正解と予測を比較し、モデルの固有表現抽出の性能を評価する。
    type_idがNoneのときは、全ての固有表現のタイプに対して評価する。
    type_idが整数を指定すると、その固有表現のタイプのIDに対して評価を行う。
    """
    num_entities = 0  # 固有表現(正解)の個数
    num_predictions = 0  # BERTにより予測された固有表現の個数
    num_correct = 0  # BERTにより予測のうち正解であった固有表現の数

    # それぞれの文章で予測と正解を比較。
    # 予測は文章中の位置とタイプIDが一致すれば正解とみなす。
    for entities, entities_predicted \
            in zip(entities_list, entities_predicted_list):

        if type_id:
            entities = [e for e in entities if e['type_id'] == type_id]
            entities_predicted = [
                e for e in entities_predicted if e['type_id'] == type_id
            ]

        get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
        set_entities = set(get_span_type(e) for e in entities)
        set_entities_predicted = set(get_span_type(e) for e in entities_predicted)

        num_entities += len(entities)
        num_predictions += len(entities_predicted)
        num_correct += len(set_entities & set_entities_predicted)

        p = num_correct / num_predictions if num_predictions != 0 else 0

        r = num_correct / num_entities if num_entities != 0 else 0
        try:
            f_scores.append(2 * p * r / (p + r))  # F値
        except Exception:
            f_scores.append(0)

    # 指標を計算
    precision = None
    recall = None
    f_value = None
    if num_predictions != 0:
        precision = num_correct / num_predictions  # 適合率
    if num_entities != 0:
        recall = num_correct / num_entities  # 再現率
    if precision != None and recall != None and precision + recall != 0:
        f_value = 2 * precision * recall / (precision + recall)  # F値

    result = {
        'num_entities': num_entities,
        'num_predictions': num_predictions,
        'num_correct': num_correct,
        'precision': precision,
        'recall': recall,
        'f_value': f_value
    }

    return result


def load_json(filepath):
    tf = open(filepath, "r")
    dataset = json.load(tf)
    return dataset


def evaluate(model_path, test_dataset_path):
    dataset_test = load_json(test_dataset_path)

    with open("id_to_tags.pkl", "rb") as tf:
        id_to_tags = pickle.load(tf)

    len_num_entity_type = len(id_to_tags)

    model = ner.BertForTokenClassification_pl.from_pretrained_bin(model_path=model_path, num_labels=2 * len_num_entity_type + 1)

    # model = ner.BertForTokenClassification_pl.load_from_checkpoint(
    #     checkpoint_path=model_path,
    # )
    bert_tc = model.bert_tc.to(device)
    tokenizer = ner.NER_tokenizer_BIO.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking',
        num_entity_type=len(id_to_tags)
    )

    entities_list = []  # 正解の固有表現を追加していく
    entities_predicted_list = []  # 抽出された固有表現を追加していく
    text_entities = []
    for sample in tqdm(dataset_test, desc='Evaluation'):
        text = sample['text']
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

        entities_list.append(sample['entities'])
        entities_predicted_list.append(entities_predicted)
        text_entities.append({'text': text, 'entities': sample['entities'], 'entities_predicted': entities_predicted})

    out = evaluate_model(entities_list, entities_predicted_list)
    print(json.dumps(out, indent=4))

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model against a test dataset')
    parser.add_argument('-m', '--model_path', type=str, default='pytorch_model.bin', help='Path to model checkpoint')
    parser.add_argument('-i', '--input', type=str, default='evaluate_data/MedTxt-CR-JA-test.json', help='Input test dataset path')

    args = parser.parse_args()

    evaluate(args.model_path, args.input)
