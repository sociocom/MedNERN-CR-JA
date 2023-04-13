---
language:
  - ja
license:
  - cc-by-4.0
tags:
  - NER
  - medical documents
datasets:
  - MedTxt-CR-JA-training-v2.xml
metrics:
  - NTCIR-16 Real-MedNLP subtask 1
---


This is a model for named entity recognition of Japanese medical documents.

### How to use

Download the following five files and put into the same folder.

- id_to_tags.pkl
- key_attr.pkl
- NER_medNLP.py
- predict.py
- text.txt (This is an input file which should be predicted, which could be changed.)

You can use this model by running `predict.py`.

```
python3 predict.py
```

#### Entity normalization

This model supports entity normalization via dictionary matching. The dictionary is a list of medical terms or
drugs and their standard forms.

Two different dictionaries are used for drug and disease normalization, stored in the `dictionaries` folder as
`drug_dict.csv` and `disease_dict.csv`, respectively.

To enable normalization you can add the `--normalize` flag to the `predict.py` command. 

```
python3 predict.py --normalize
```

Normalization will add the `norm` attribute to the output XML tags. This attribute can be empty if a normalized form of
the term is not found.

The provided disease normalization dictionary (`dictionaties/disease_dict.csv`) is based on the [Manbyo Dictionary](https://sociocom.naist.jp/manbyo-dic-en/) and provides normalization to the standard ICD code for the diseases.

The default drug dictionary (`dictionaties/drug_dict.csv`) is based on the [Hyakuyaku Dictionary](https://sociocom.naist.jp/hyakuyaku-dic-en/).

The dictionary is a CSV file with three columns: the first column is the surface form term and the third column contain
its standard form. The second column is not used.

User can freely change the dictionary to fit their needs, as long as the format and filename are kept.

### Input Example

```
肥大型心筋症、心房細動に対してＷＦ投与が開始となった。
治療経過中に非持続性心室頻拍が認められたためアミオダロンが併用となった。
```

### Output Example

```
<d certainty="positive" norm="I422">肥大型心筋症、心房細動</d>に対して<m-key state="executed" norm="ワルファリンカリウム">ＷＦ</m-key>投与が開始となった。
<timex3 type="med">治療経過中</timex3>に<d certainty="positive" norm="I472">非持続性心室頻拍</d>が認められたため<m-key state="executed" norm="アミオダロン塩酸塩">アミオダロン</m-key>が併用となった。
```

### Publication

