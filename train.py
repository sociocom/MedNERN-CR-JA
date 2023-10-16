import argparse
import json
import os
import pickle
import random

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import NER_medNLP as ner
# %%
import from_XML_to_json as XtC
from evaluation import evaluate
from predict import run as predict


# %%
# データセットの分割(8:2に分割)
def load_json(filepath):
    tf = open(filepath, "r")
    dataset = json.load(tf)

    n = len(dataset)
    n_train = int(n * 0.8)
    dataset_train = random.sample(dataset, n_train)
    dataset_val = list(filter(lambda i: i not in dataset_train, dataset))
    return dataset_train, dataset_val


# %%
def create_dataset(tokenizer, dataset, max_length):
    """
    データセットをデータローダに入力できる形に整形。
    """
    dataset_for_loader = []
    for sample in dataset:
        text = sample['text']
        entities = sample['entities']
        encoding = tokenizer.encode_plus_tagged(
            text, entities, max_length=max_length
        )
        # print(encoding)
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        dataset_for_loader.append(encoding)
    return dataset_for_loader


# %%
def training(model_name: str, output_folder: str, dataset_path: str, max_epochs: int, lr: float, batch_size: int, from_scratch: bool = False):
    dataset_train, dataset_val = load_json(dataset_path)

    frequent_tags_attrs, _, _ = XtC.select_tags(attrs=True)  # タグ取得

    # id_to_tags.pklを作成
    id_to_tags = {i + 1: v for i, v in enumerate(frequent_tags_attrs)}
    pickle.dump(id_to_tags, open('id_to_tags.pkl', 'wb'))

    tokenizer = ner.NER_tokenizer_BIO.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking',
        num_entity_type=len(id_to_tags)
    )

    # データセットの作成
    max_length = 512
    dataset_train_for_loader = create_dataset(
        tokenizer, dataset_train, max_length
    )
    dataset_val_for_loader = create_dataset(
        tokenizer, dataset_val, max_length
    )

    # データローダの作成
    dataloader_train = DataLoader(
        dataset_train_for_loader, batch_size=batch_size, shuffle=True
    )
    dataloader_val = DataLoader(dataset_val_for_loader, batch_size=batch_size)

    # ファインチューニング
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath='model_BIO/'
    )

    trainer = pl.Trainer(
        devices=1, accelerator="gpu",
        max_epochs=max_epochs,
        callbacks=[checkpoint]
    )

    num_entity_type = len(frequent_tags_attrs)
    num_labels = 2 * num_entity_type + 1

    if args.from_scratch:
        print("Disregarding the pre-trained sociocom/MedNERN-CR-JA weights and using the base model directly")
        model = ner.BertForTokenClassification_pl(num_labels=num_labels, lr=lr, model='cl-tohoku/bert-base-japanese-whole-word-masking')
    else:
        model = ner.BertForTokenClassification_pl(num_labels=num_labels, lr=lr)

    model.configure_optimizers()

    trainer.fit(model, dataloader_train, dataloader_val)
    # os.makedirs('trained_models', exist_ok=True)
    # trainer.save_checkpoint('trained_models/' + model_name + ".ckpt")
    os.makedirs(output_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_folder, '{}.bin'.format(model_name)))
    return model


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('-m', '--model_path', type=str, default='pytorch_model', help='The name to be given to the model file')
    parser.add_argument('-d', '--training_dataset', type=str, default='training_data/MedTxt-CR-JA-training.json', help='Path to the training dataset in JSON format')
    parser.add_argument('-o', '--output', type=str, default='models', help='Path to output the trained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate to be used in training')
    parser.add_argument('--batch_size', type=float, default=32, help='Batch size')
    parser.add_argument('--from-scratch', action=argparse.BooleanOptionalAction, help='Used to disregard the pre-trained sociocom/MedNERN-CR-JA weights and use the base model directly', default=False)

    # Evaluation and test parameters
    parser.add_argument('-t', '--test_dataset', type=str, default=None,
                        help='Path to the test dataset in JSON format for evaluation of the model (if not provided, evaluation will be skipped)', required=False)
    parser.add_argument('-f', '--test_file', type=str, default='text.txt', help='Path to a example text file to predict using the trained model', required=False)

    args = parser.parse_args()

    model_name = args.model_name
    model_path = os.path.join(args.output, '{}.bin'.format(model_name))

    # Train
    model = training(model_name, args.output, args.training_dataset, args.epochs, args.lr, args.batch_size, args.from_scratch)

    # Evaluate
    if args.test_dataset:
        evaluate(model_path, args.test_dataset)

    # Predict
    predict(model_path, args.test_file)
