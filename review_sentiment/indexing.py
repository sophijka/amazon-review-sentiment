import logging
import sys
import os
import pandas as pd
import re
from os import walk
import json
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from expertai.nlapi.cloud.client import ExpertAiClient
from transformers import MarianMTModel, MarianTokenizer
from pysbd.utils import PySBDFactory
import spacy
import torch
import os

os.environ["EAI_USERNAME"] = "sophijka@yahoo.com"
os.environ["EAI_PASSWORD"] = "Osin1Lwiw$%"


class Indexer:

    def __init__(self, host, port, username, password, index, language, logger_level = logging.WARNING):
        """
        :param config: dictionary from yaml config
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.index = index
        self.language = language
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logger_level)

        self.document_store = ElasticsearchDocumentStore(host=self.host,
                                                         port=self.port,
                                                         username=self.username,
                                                         password=self.password,
                                                         index=self.index,
                                                         analyzer=self.language)
        self.expertai_client = ExpertAiClient()
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.mt_model_name = f'Helsinki-NLP/opus-mt-de-en'
        self.mt_tokenizer = MarianTokenizer.from_pretrained(self.mt_model_name)
        self.mt_model = MarianMTModel.from_pretrained(self.mt_model_name).to(self.torch_device)

    def preprocess_reviews(self, review_file):
        """
        Reading files to index and split them into paragraphs, if true
        :return: a listvfvfv of Python dictionaries with text (or paragraphs), in "text", and metadata in "meta"
        """

        data = []
        # nlp = spacy.blank('de')
        # nlp.add_pipe(PySBDFactory(nlp))
        identifier = 0
        with open(review_file) as f:
            for line in f:
                # data.append(json.loads(line))
                d = json.loads(line)
                meta = {}
                review = {}
                meta['review_id'] = d['review_id']
                meta['product_id'] = d['product_id']
                meta['reviewer_id'] = d['reviewer_id']
                meta['stars'] = d['stars']
                meta['review_title'] = d['review_title']
                meta['language'] = d['language']
                meta['product_category'] = d['product_category']
                # review_text = d['review_body']
                if d['language'] == 'en':
                    review_text = d['review_body']
                elif d['language'] == 'de':
                    review_text = ' '.join([str(elem) for elem in self.translate(d['review_body'])])
                    review['original_text'] = d['review_body']
                    print("translated:", review_text)

                review_sent = self.review_sentiment(review_text)
                meta['sentiment'] = review_sent.sentiment.overall

                knowledge_syncons = ["product.commodity", "artifact.instrument", "object.food", "food.beverage"]
                for i in review_sent.knowledge:
                    if i.label in knowledge_syncons:
                        print("i.syncon", i.syncon)
                        for j in review_sent.sentiment.items:
                            if j.syncon == i.syncon:
                                # replace a dot with underscore, for Elastic
                                label = i.label.replace(".", "_")
                                meta[label] = j.lemma
                                print("i.label", i.label)
                                print("i.lemma", j.lemma)

                if meta['sentiment'] is not None:
                    meta['id'] = identifier
                    review['text'] = review_text
                    review['meta'] = meta
                    data.append(review)
                    identifier = identifier + 1

        print(data)
        return data

    # def sentence_detection(self, texts):
    #     passage = self.nlp(texts)
    #     return list(passage.sents)

    def translate(self, stexts, language="de", sb=False):
        # def translate(stexts, model, tokenizer, language="nld", sb=False):
        # Prepare the text data into appropriate format for the model

        # sentence boundary detection
        list_src = []
        src_texts = f">>{language}<< {stexts}"
        list_src.append(src_texts)

        # if sb:
        #     texts = self.setence_detection(stexts)
        #     template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
        #     list_src = [template(text) for text in texts]
        #
        # else:
        #     src_texts = f">>{language}<< {stexts}"
        #     list_src.append(src_texts)

        translated = self.mt_model.generate(
                **self.mt_tokenizer.prepare_seq2seq_batch(list_src, return_tensors="pt").to(self.torch_device))
        # convert generated token indices into text
        translated_texts = [self.mt_tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        return translated_texts

    def review_sentiment(self, text, language='en'):
        print("input", text)
        output = None
        try:
            output = self.expertai_client.specific_resource_analysis(
             body={"document": {"text": text}}, params={'language': language, 'resource': 'sentiment'})
        except:
            pass
        # print(output.knowledge)
        return output


    def index_data(self, review_path):
        self.document_store.write_documents(self.preprocess_reviews(review_path))

    def data_stats(self):
        return self.document_store.describe_documents(self.index)

    def delete_index(self):
        self.document_store.delete_all_documents(self.index)

    def delete_entry(self, val, attribute):

        deleting = {
            "query": {
                "match": {
                    attribute: val
                }
            }
        }
        self.document_store.client.delete_by_query(index=self.index, body=deleting, ignore=[404])
