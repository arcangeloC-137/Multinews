import re
import nltk
import spacy
import torch
import datasets
import logging
import pandas as pd

from datasets import Dataset
from datasets import load_metric
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

nltk.download('punkt')

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)


class ContextGenerator():

    def __init__(self,
                 ds_train_name=None,
                 ds_test_name=None,
                 base_model_name="allenai/led-base-16384",
                 epochs=15,
                 batch_size=3,
                 max_length=4096,
                 data_path=None,
                 output_dir=None,
                 device=None):

        self.results = None
        self.rouge = None
        self.train_data = None
        self.eval_data = None
        self.test_data = None

        logging.info("Context Generator - Initialization")

        self.epochs = epochs
        self.batch_size = batch_size


        self.df_train_raw = None
        self.df_test_raw = None

        self.ds_train_name = ds_train_name
        self.ds_test_name = ds_test_name
        self.base_model_name = base_model_name
        self.max_length = max_length
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Model loading
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_name,
                                                           gradient_checkpointing=True,
                                                           use_cache=False)
        self.model.gradient_checkpointing_enable()
        
    
    def extract_context(self, data_path=None, ds_train_name=None,ds_test_name=None, output_dir=None):
    
        self.data_path = data_path
        self.ds_train_name = ds_train_name
        self.ds_test_name = ds_test_name
        self.output_dir = output_dir
        
        self.load_data()

        train_context_tuples = self.preprocess_data(self.df_train_raw[:150])
        test_context_tuples = self.preprocess_data(self.df_test_raw[:80])

        df_train = pd.DataFrame(list(train_context_tuples), columns=['text', 'summary'])
        df_test = pd.DataFrame(list(test_context_tuples), columns=['text', 'summary'])

        self.prepare_data(df_train, df_test)
        self.train()
        self.save_context(self.results)

    def load_data(self):

        self.df_train_raw = pd.read_json(self.data_path+self.ds_train_name)
        self.df_test_raw = pd.read_json(self.data_path+self.ds_test_name)

    def prepare_data(self, df_train, df_test):

        df_train, df_eval = train_test_split(df_train, test_size=0.2, random_state=42)

        df_train['summary'] = df_train['summary'].map(self.list_to_string)
        df_eval['summary'] = df_eval['summary'].map(self.list_to_string)
        df_test['summary'] = df_test['summary'].map(self.list_to_string)

        self.train_data = datasets.DatasetDict({"train": Dataset.from_dict(df_train)})
        self.eval_data = datasets.DatasetDict({"train": Dataset.from_dict(df_eval)})
        self.test_data = datasets.DatasetDict({"train": Dataset.from_dict(df_test)})

    def prepare_data_for_model(self, batch):

        max_input_length = self.max_length

        # BERT-base max input lenght
        max_output_length = 512

        inputs = self.tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_input_length,
        )
        outputs = self.tokenizer(
            batch["summary"],
            padding="max_length",
            truncation=True,
            max_length=max_output_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch

    def train(self):

        # Model parameters
        self.model.config.num_beams = 2
        self.model.config.max_length = 512
        self.model.config.min_length = 100
        self.model.config.length_penalty = 2.0
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3

        df_train = self.train_data.map(
            self.prepare_data_for_model,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["text", "summary"],
        )

        df_eval = self.eval_data.map(
            self.prepare_data_for_model,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["text", "summary"],
        )

        df_train.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        df_eval.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        self.rouge = load_metric("rouge")

        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            fp16=True,  # Whether to use 16-bit (mixed) precision training instead of 32-bit training.
            output_dir=self.output_dir,
            warmup_steps=10,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir=self.output_dir + '/logs',  # directory for storing logs
            logging_steps=10,
            logging_first_step=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            gradient_accumulation_steps=4,
            num_train_epochs=self.epochs
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=df_train["train"],
            eval_dataset=df_eval["train"],
        )

        trainer.train()
        self.save_model(self.model)
        
        self.results = self.test_data.map(self.generate_context, batched=True, batch_size=self.batch_size)

        logging.info("Result:", self.rouge.compute(
            predictions=self.results['train']["predicted_abstract"],
            references=self.results['train']["summary"],
            rouge_types=["rouge2"])["rouge2"].mid)

    def list_to_string(self, ls):
        res = ''
        for sentence in ls:
            res += ' ' + sentence
        return res

    def preprocess_data(self, df):

        context_tuples = list()
        ctx_avg_lenght = 0

        for index, row in df.iterrows():
            context = ""

            if 'ABSTRACT' in row['papers']['sections'].keys():
                context = "".join([context, row['papers']['sections']['ABSTRACT']])

            if 'INTRODUCTION' in row['papers']['sections'].keys():
                context = "".join([context, row['papers']['sections']['INTRODUCTION']])

            if 'RESULTS' in row['papers']['sections'].keys():
                context = "".join([context, row['papers']['sections']['RESULTS']])
            context = self.clean_text(context)
            context = self.clean_sentence(context)
            context_tuples.append((context, row['papers']['highlights']))
            ctx_avg_lenght += len(context)

        logging.info(f'Context average lenght: {ctx_avg_lenght // len(context_tuples)}')

        return context_tuples

    def clean_text(self, text):
        clean = text.replace('\n', ' ')
        clean = re.sub(' +', ' ', clean)
        return clean

    def clean_sentence(self, s, remove_punct=True, remove_sym=True, remove_stop=True):

        nlp = spacy.load("en_core_web_lg", disable=['ner'])
        analyzed_sentence = nlp(s)
        clean_token = []

        for token in analyzed_sentence:
            if token.pos_ != "PUNCT":
                clean_token.append(token)

        if remove_punct:
            ct = []
            for token in clean_token:
                if token.pos_ != "PUNCT":
                    ct.append(token)
            clean_token = ct

        if remove_sym:
            ct = []
            for token in clean_token:
                if not token.is_stop:
                    ct.append(token)
            clean_token = ct

        if remove_stop:
            ct = []
            for token in clean_token:
                if token.pos_ != "SYM":
                    ct.append(token)
            clean_token = ct

        return ' '.join(word.text for word in clean_token)

    def compute_metrics(self, pred):

        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output_2 = self.rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        rouge_output_1 = self.rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge1"]
        )["rouge1"].mid

        rouge_output_l = self.rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rougeL"]
        )["rougeL"].mid

        return {
            "rouge1_precision": round(rouge_output_1.precision, 4),
            "rouge1_recall": round(rouge_output_1.recall, 4),
            "rouge1_fmeasure": round(rouge_output_1.fmeasure, 4),
            "rouge2_precision": round(rouge_output_2.precision, 4),
            "rouge2_recall": round(rouge_output_2.recall, 4),
            "rouge2_fmeasure": round(rouge_output_2.fmeasure, 4),
            "rougeL_precision": round(rouge_output_l.precision, 4),
            "rougeL_recall": round(rouge_output_l.recall, 4),
            "rougeL_fmeasure": round(rouge_output_l.fmeasure, 4),
        }

    def save_model(self, model):
        model.save_pretrained(self.output_dir + '/model')

    def generate_context(self, batch):
        inputs_dict = self.tokenizer(batch["text"], padding="max_length", max_length=512, return_tensors="pt",
                                     truncation=True)
        input_ids = inputs_dict.input_ids.to("cuda")
        attention_mask = inputs_dict.attention_mask.to("cuda")
        global_attention_mask = torch.zeros_like(attention_mask)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1

        predicted_abstract_ids = self.model.generate(input_ids, attention_mask=attention_mask,
                                                     global_attention_mask=global_attention_mask)
        batch["predicted_abstract"] = self.tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
        return batch

    def save_context(self, results):
        res_dict = {}

        for idx, abstract in enumerate(results['train']['predicted_abstract']):
            tokenized_pred = nltk.sent_tokenize(abstract)
            tokenized_abs = nltk.sent_tokenize(results['train']['summary'][idx])

            res_dict[idx] = {'prediction': tokenized_pred, 'abstract': tokenized_abs}

        res_df = pd.DataFrame(res_dict)
        res_df.T.to_csv(self.output_dir+'/extracted_context')
