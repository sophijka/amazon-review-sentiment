import logging
import sys
import os
import pandas as pd
import re
from os import walk
import json
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore


def find(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    return -1


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


    def preprocess_reviews(self, review_file):
        """
        Reading files to index and split them into paragraphs, if true
        :return: a listvfvfv of Python dictionaries with text (or paragraphs), in "text", and metadata in "meta"
        """

        data = []

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
                review['text'] = d['review_body']
                review['meta'] = meta
                data.append(review)

        print(data)
        return data

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
