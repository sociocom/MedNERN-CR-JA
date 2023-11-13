
This is a model for named entity recognition of Japanese medical documents.

# Introduction

This repository contains scripts for training and using the model.

It is intended for those who want to re-train/fine-tune the model with additional or different datasets. If you just want to use the model, you should follow the instructions in the [HuggingFace model page](https://huggingface.co/sociocom/MedNERN-CR-JA).

The original model was trained on the [MedTxt-CR-JA](https://sociocom.naist.jp/medtxt/cr) dataset, so the code expects and outputs XML tags in the same format.

## How to use

Install the requirements:

``` 
pip install -r requirements.txt
```

The model can be trained, evaluated and used for prediction using the scripts `train.py`, `evaluate.py` and `predict.py`, respectively.

The code has been developed tested with Python 3.9 in MacOS 14.1 (M1 MacBook Pro).

## Training

In order to feed the dataset to the model, it must first be converted from the original XML format into JSON that is
parseable by the model.
To do so, run the following command:

```
python3 from_XML_to_json.py -i <your_dataset_file>.xml
```

This will create a file called `<your_dataset_file>.json` in the same folder as the dataset file.
Then the model can be trained by running the following command:

```
python3 train.py -d <your_dataset_file>.json
```

At least `-d` (training dataset) parameter must be provided.
The model name is used to create a folder in the output directory (root folder is the default, can be changed by parameter) where the model will be saved.

The training script accepts additional arguments, for instance, to specify the number of epochs, the batch size, etc.

Also, if a test dataset is provided by the `-t` parameter, the model will be evaluated after the training is finished
and the results will be printed to the screen.

Be aware that the training script will overwrite any existing model with the same name and also overwrite
the `id_to_tags.pkl` file, used for converting model ids to the output classes.

## Evaluation

The evaluation script will run the provided model on the test dataset and print the results to the screen.
It can be run with the following command:

```
python3 evaluate.py -m <model_path> -i <your_dataset_file>.json
```

## Prediction

The prediction script will output the results in the same XML format as the input file. It can be run with the following
command:

```
python3 predict.py
```

The default parameters will take the model located in `pytorch_model.bin` and the input file `text.txt`.
The resulting predictions will be output to the screen.

To select a different model or input file, use the `-m` and `-i` parameters, respectively:

```
python3 predict.py -m <model_path> -i <your_input_file>.txt
```

The input file can be a single text file or a folder containing multiple `.txt` files, for batch processing. For example:

```
python3 predict.py -m <model_path> -i <your_input_folder>
```


### Entity normalization

This model supports entity normalization via dictionary matching. The dictionary is a list of medical terms or
drugs and their standard forms.

Two different dictionaries are used for drug and disease normalization, stored in the `dictionaries` folder as
`drug_dict.csv` and `disease_dict.csv`, respectively.

To enable normalization you can add the `--normalize` flag to the `predict.py` command.

```
python3 predict.py -m <model_path> --normalize
```

Normalization will add the `norm` attribute to the output XML tags. This attribute can be empty if a normalized form of
the term is not found.

The provided disease normalization dictionary (`dictionaties/disease_dict.csv`) is based on
the [Manbyo Dictionary](https://sociocom.naist.jp/manbyo-dic-en/) and provides normalization to the standard ICD code
for the diseases.

The default drug dictionary (`dictionaties/drug_dict.csv`) is based on
the [Hyakuyaku Dictionary](https://sociocom.naist.jp/hyakuyaku-dic-en/).

The dictionary is a CSV file with three columns: the first column is the surface form term and the third column contain
its standard form. The second column is not used.

### Replacing the default dictionaries

User can freely change the dictionary to fit their needs by passing the path to a custom dictionary file.
The dictionary file must have at least a column containing the list of surface forms and a column containing the list of
normalized forms.

The parameters `--drug_dict` and `--disease_dict` can be used to specify the path to the drug and disease dictionaries,
respectively.
When doing so, the respective parameters informing the column index of the surface form and normalized form must also be
provided.
You don't need to replace both dictionaries at the same time, you can replace only one of them.

E.g.:

```
python3 predict.py --normalize --drug_dict dictionaries/drug_dict.csv --drug_surface_form 0 --drug_norm_form 2 --disease_dict dictionaries/disease_dict.csv --disease_surface_form 0 --disease_norm_form 2
```

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

## Publication

This model can be cited as:

```
@misc {social_computing_lab_2023,
	author       = { {Social Computing Lab} },
	title        = { MedNERN-CR-JA (Revision 13dbcb6) },
	year         = 2023,
	url          = { https://huggingface.co/sociocom/MedNERN-CR-JA },
	doi          = { 10.57967/hf/0620 },
	publisher    = { Hugging Face }
}
```
