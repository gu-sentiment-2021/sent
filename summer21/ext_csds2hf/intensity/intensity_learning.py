import transformers
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer, \
    DataCollatorWithPadding
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from copy import deepcopy
import os
import json
import random
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from random import randrange
from random import seed
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import nlpaug
import nlpaug.augmenter.word as naw


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class IntensityLearning:
    model_name = 'bert-base-cased'

    tokenizer = None
    device = None

    mapping_dict = {'high-extreme': 6, 'extreme': 5, 'medium': 2, 'medium-high': 3, 'low': 0, 'high': 4,
                    'low-medium': 1}
    classes_counts = {'high-extreme': 0, 'extreme': 0, 'medium': 0, 'medium-high': 0, 'low': 0, 'high': 0,
                      'low-medium': 0}

    X = []
    y = []
    path = 'data.json'
    aug_path = ''
    TOPK = 20  # default=100
    ACT = 'insert'  # "substitute"
    aug_bert = None
    RANDOM_SEED = 42
    MAX_LENGTH = 256

    def __init__(self, path='data.json', aug_path='', model_name='bert-base-cased'):
        self.path = path
        self.aug_path = aug_path
        self.model_name = model_name

        ## Define pretrained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        aug_bert = naw.ContextualWordEmbsAug(
            model_path=model_name,  # 'distilbert-base-uncased',
            # device='cuda',
            action=self.ACT, top_k=self.TOPK)

    def load_json_file(self):
        # Loading the saved JSON file.
        with open(self.path) as json_file:
            data = json.load(json_file)
        return data

    def random_shuffler(self, a, b):
        c = list(zip(a, b))
        random.shuffle(c)
        a, b = zip(*c)
        return a, b

    def dataset_shuffler(self, a, b, num_iterations=1000):
        i = 0
        while i < num_iterations:
            a, b = self.random_shuffler(a, b)
            i += 1
        return a, b

    def collect_data(self, data):
        for item in data['csds_objects']:
            if item['intensity']:
                if self.mapping_dict[item['intensity']] > -1:
                    self.X.append([item['text'], item['head']])
                    self.y.append(self.mapping_dict[item['intensity']])
                    self.classes_counts[item['intensity']] += 1

    def augment_add(self):

        for cl in self.mapping_dict.keys():
            if self.mapping_dict[cl] > -1:
                # Loading the saved JSON file.
                with open(self.aug_path + cl + '_data_aug.json') as json_file:
                    data_aug = json.load(json_file)

                x_data = data_aug['X']
                y_data = data_aug['y']
                for i in range(len(x_data)):
                    self.X.append([x_data[i][0], x_data[i][1]])
                    self.y.append(self.mapping_dict[y_data[i]])
                    self.classes_counts[y_data[i]] += 1

    def get_possible_classes(self):
        possible_classes = set()
        possible_classes_labels = set()

        for k in self.mapping_dict.keys():
            if self.mapping_dict[k] > -1:
                possible_classes.add(self.mapping_dict[k])
                possible_classes_labels.add(k)
        return possible_classes, possible_classes_labels

    def get_training_arg(self):
        training_args = TrainingArguments(
            # The output directory where the model predictions
            # and checkpoints will be written.
            output_dir='pretrain_bert',

            # Overwrite the content of the output directory.
            overwrite_output_dir=True,

            # Whether to run training or not.
            do_train=True,

            # Whether to run evaluation on the dev or not.
            do_eval=True,

            # NEW
            # seed=self.RANDOM_SEED,

            # Batch size GPU/TPU core/CPU training.
            per_device_train_batch_size=10,

            # Batch size  GPU/TPU core/CPU for evaluation.
            per_device_eval_batch_size=100,

            # evaluation strategy to adopt during training
            # `no`: No evaluation during training.
            # `steps`: Evaluate every `eval_steps`.
            # `epoch`: Evaluate every end of epoch.
            evaluation_strategy='steps',

            # How often to show logs. I will se this to
            # plot history loss and calculate perplexity.
            logging_steps=100,

            # Number of update steps between two
            # evaluations if evaluation_strategy="steps".
            # Will default to the same value as l
            # logging_steps if not set.
            eval_steps=None,

            # Set prediction loss to `True` in order to
            # return loss for perplexity calculation.
            prediction_loss_only=False,

            # The initial learning rate for Adam.
            # Defaults to 5e-5.
            learning_rate=5e-5,

            # The weight decay to apply (if not zero).
            weight_decay=0,

            # Epsilon for the Adam optimizer.
            # Defaults to 1e-8
            adam_epsilon=1e-8,

            # Maximum gradient norm (for gradient
            # clipping). Defaults to 0.
            max_grad_norm=1.0,
            # Total number of training epochs to perform
            # (if not an integer, will perform the
            # decimal part percents of
            # the last epoch before stopping training).
            num_train_epochs=4,

            # Number of updates steps before two checkpoint saves.
            # Defaults to 500
            save_steps=-1,
        )
        return training_args

    # Define Trainer parameters
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def print_performance_measures(self, results):
        # Extract metrics
        acc = results['eval_accuracy']
        recall = results['eval_recall']
        precision = results['eval_precision']
        f1 = results['eval_f1']
        # Show metrics
        print(f'Accuracy: {acc}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')

    def show_confusion_matrix(self, confusion_matrix, possible_classes=None, save_path=''):
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=cmap)  # "YlGnBu" #"Blues"
        if possible_classes == None:
            hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
            hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        else:
            # We want to show all ticks...
            hmap.set_xticks(np.arange(len(possible_classes)) + 0.5)
            hmap.set_yticks(np.arange(len(possible_classes)) + 0.5)
            # ... and label them with the respective list entries
            hmap.set_xticklabels(possible_classes)
            hmap.set_yticklabels(possible_classes)

            # Rotate the tick labels and set their alignment.
            plt.setp(hmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.title('Confusion Matrix')
            plt.savefig(save_path + 'confusion_matrix.png', dpi=700, bbox_inches='tight')

        plt.ylabel('True')
        plt.xlabel('Predicted')

    def k_fold_cross_validation_train(self, num_folds, possible_classes):
        X = np.array(self.X)
        y = np.array(self.y)
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=self.RANDOM_SEED)
        skf.get_n_splits(X, y)
        # Performance measures
        acc = 0
        recall = None
        precision = None
        f1 = None
        fold_ite = 1
        loss = []
        # Training
        for train_index, test_index in skf.split(X, y):
            print(f'<<<<<<<< Fold #{fold_ite} >>>>>>>>')
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_tr, X_va = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]
            #
            y_train = y_train.tolist()
            y_val = y_val.tolist()
            X_train = []
            X_p_train = []
            for item in X_tr:
                X_train.append(item[0])
                X_p_train.append(item[1])

            X_val = []
            X_p_val = []
            for item in X_va:
                X_val.append(item[0])
                X_p_val.append(item[1])

            ###
            num_possible_classes = len(possible_classes)
            ###
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_possible_classes)
            X_train_tokenized = self.tokenizer(X_train, X_p_train, padding=True, truncation=True,
                                               max_length=self.MAX_LENGTH)
            X_val_tokenized = self.tokenizer(X_val, X_p_val, padding=True, truncation=True, max_length=self.MAX_LENGTH)
            train_dataset = Dataset(X_train_tokenized, y_train)
            val_dataset = Dataset(X_val_tokenized, y_val)
            training_args = self.get_training_arg()
            #
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
            )
            #
            print("Starting training")
            trainer.train()
            print("Done training")
            results = trainer.evaluate()
            print(results.keys())
            if fold_ite == 1:
                acc = results['eval_accuracy']
                recall = results['eval_recall']
                precision = results['eval_precision']
                f1 = results['eval_f1']
            else:
                acc += results['eval_accuracy']
                recall += results['eval_recall']
                precision += results['eval_precision']
                f1 += results['eval_f1']

            loss.append(results['eval_loss'])
            fold_ite += 1

        # Show metrics
        print(f'Accuracy: {acc / num_folds}')
        print(f'Precision: {precision / num_folds}')
        print(f'Recall: {recall / num_folds}')
        print(f'F1: {f1 / num_folds}')
        # Get predictions
        encoded_input = self.tokenizer(X_val, X_p_val, max_length=self.MAX_LENGTH, padding=True, truncation=True,
                                       return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input.to(self.device))

        logits = model_output.logits
        probs = F.softmax(logits, dim=-1)
        y_pred = torch.argmax(probs, dim=1).cpu().numpy()

        # Show confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        df_cm = pd.DataFrame(cm, index=possible_classes, columns=possible_classes)
        self.show_confusion_matrix(df_cm)
        return model

    def train_test_split_train(self, possible_classes, test_size=0.33, show_labels=False):
        X_tr, X_va, y_train, y_val = train_test_split(self.X, self.y, test_size=test_size, random_state=42,
                                                      shuffle=True,
                                                      stratify=self.y)
        X_train = []
        X_p_train = []
        for item in X_tr:
            X_train.append(item[0])
            X_p_train.append(item[1])

        X_val = []
        X_p_val = []
        for item in X_va:
            X_val.append(item[0])
            X_p_val.append(item[1])

        ###
        num_possible_classes = len(possible_classes)
        ###
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_possible_classes)
        X_train_tokenized = self.tokenizer(X_train, X_p_train, padding=True, truncation=True,
                                           max_length=self.MAX_LENGTH)
        X_val_tokenized = self.tokenizer(X_val, X_p_val, padding=True, truncation=True, max_length=self.MAX_LENGTH)
        train_dataset = Dataset(X_train_tokenized, y_train)
        val_dataset = Dataset(X_val_tokenized, y_val)
        training_args = self.get_training_arg()
        #
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        #
        print("Starting training")
        trainer.train()
        print("Done training")
        results = trainer.evaluate()
        self.print_performance_measures(results)
        ###
        del X_train_tokenized, X_val_tokenized
        del train_dataset, val_dataset
        del X_train, X_p_train
        # Get predictions
        encoded_input = self.tokenizer(X_val, X_p_val, max_length=self.MAX_LENGTH, padding=True, truncation=True,
                                       return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input.to(self.device))

        logits = model_output.logits
        probs = F.softmax(logits, dim=-1)
        y_pred = torch.argmax(probs, dim=1).cpu().numpy()

        # Show confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        df_cm = pd.DataFrame(cm, index=possible_classes, columns=possible_classes)
        if show_labels:
            self.show_confusion_matrix(df_cm, possible_classes=possible_classes)
        else:
            self.show_confusion_matrix(df_cm)
        return model

    def save_model_state(self, model, model_name, path=''):
        model_save_name = model_name + '.pt'
        save_path = f"{path}/{model_save_name}"
        torch.save(model.state_dict(), save_path)

    def load_model_state(self, model, model_name, path=''):
        model_save_name = model_name + '.pt'
        load_path = f"{path}/{model_save_name}"
        model.load_state_dict(torch.load(load_path))
        return model

##'xlm-roberta-base' #'albert-base-v1' #'albert-base-v2' #'roberta-large' #'distilbert-base-cased' #"bert-base-cased"  #'deepset/roberta-base-squad2' #"bert-base-uncased"  #'bert-base-multilingual-cased'
##'google/bert_uncased_L-2_H-128_A-2' 'google/bert_uncased_L-8_H-512_A-8'

# n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)
