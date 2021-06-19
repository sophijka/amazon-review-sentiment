import logging
import json
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from expertai.nlapi.cloud.client import ExpertAiClient
from transformers import MarianMTModel, MarianTokenizer
import torch
import os

os.environ["EAI_USERNAME"] = ""
os.environ["EAI_PASSWORD"] = ""


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

        self.mt_model_name_de = f'Helsinki-NLP/opus-mt-de-en'
        self.mt_tokenizer_de = MarianTokenizer.from_pretrained(self.mt_model_name_de)
        self.mt_model_de = MarianMTModel.from_pretrained(self.mt_model_name_de).to(self.torch_device)

        self.mt_model_name_es = f'Helsinki-NLP/opus-mt-es-en'
        self.mt_tokenizer_es = MarianTokenizer.from_pretrained(self.mt_model_name_es)
        self.mt_model_es = MarianMTModel.from_pretrained(self.mt_model_name_es).to(self.torch_device)

        self.mt_model_name_ja = f'Helsinki-NLP/opus-mt-ja-en'
        self.mt_tokenizer_ja = MarianTokenizer.from_pretrained(self.mt_model_name_ja)
        self.mt_model_ja = MarianMTModel.from_pretrained(self.mt_model_name_ja).to(self.torch_device)

    def preprocess_reviews(self, review_file):
        """
        Preprocess reviews by translating into English, if necessary, and extracting sentiment.
        :param review_file: input review file in json
        :return:
        """

        data = []
        identifier = 0
        with open(review_file) as f:
            for line in f:
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

                if d['language'] == 'en':
                    review_text = d['review_body']

                elif d['language'] == 'de':
                    review_text = ' '.join([str(elem) for elem in self.translate(d['review_body'])])
                    review['original_text'] = d['review_body']

                elif d['language'] == 'es':
                    review_text = ' '.join([str(elem) for elem in self.translate(d['review_body'], language='es')])
                    review['original_text'] = d['review_body']

                elif d['language'] == 'ja':
                    review_text = ' '.join([str(elem) for elem in self.translate(d['review_body'], language='ja')])
                    review['original_text'] = d['review_body']

                review_sent = self.review_sentiment(review_text)
                if review_sent:
                    if review_sent.sentiment.overall:
                        meta['sentiment'] = review_sent.sentiment.overall
                    if review_sent.sentiment.positivity:
                        meta['positivity'] = review_sent.sentiment.positivity
                    if review_sent.sentiment.negativity:
                        meta['negativity'] = review_sent.sentiment.negativity
                    item_sentiment = self.all_items_sentiment(review_sent.sentiment.items)
                    positive_phrases = []
                    negative_phrases = []
                    neutral_phrases = []

                    for s in item_sentiment:
                        for key in s:
                            if s[key] > 0:
                                positive_phrases.append(key)
                            elif s[key] < 0:
                                negative_phrases.append(key)
                            else:
                                neutral_phrases.append(key)

                    if positive_phrases:
                        pos = ' :'.join(positive_phrases)
                        meta['positive_phrases'] = pos
                        meta['positive_phrases'] = meta['positive_phrases'].lstrip()
                    if negative_phrases:
                        meta['negative_phrases'] = ' :'.join(negative_phrases)
                        meta['negative_phrases'] = meta['negative_phrases'].lstrip()
                    if neutral_phrases:
                        meta['neutral_phrases'] = ' :'.join(neutral_phrases)
                        meta['neutral_phrases'] = meta['neutral_phrases'].lstrip()

                    knowledge_syncons = ["product.commodity", "artifact.instrument", "object.food", "food.beverage"]
                    for i in review_sent.knowledge:
                        if i.label in knowledge_syncons:
                            self.logger.info(f"Syncon: {i.syncon}")
                            for j in review_sent.sentiment.items:
                                if j.syncon == i.syncon:
                                    # replace a dot with underscore, for Elastic
                                    label = i.label.replace(".", "_")
                                    meta[label] = j.lemma
                                    self.logger.info(f"Label: {i.label}")
                                    self.logger.info(f"Lemma: {j.lemma}")

                    # if meta['sentiment'] is not None:
                    if review_sent.sentiment.overall:
                        meta['id'] = identifier
                        review['text'] = review_text
                        review['meta'] = meta
                        data.append(review)
                        identifier = identifier + 1
            self.logger.info(f"Data {data}")
        return data

    def translate(self, stexts, language="de", sb=False):
        """
        Translate text from original language (German by default) into English
        :param stexts: text to translate
        :param language: original language
        :param sb: sentence boundary detection flag
        :return: translated text in English
        """

        list_src = []
        src_texts = f">>{language}<< {stexts}"
        list_src.append(src_texts)

        if language == 'de':
            translated = self.mt_model_de.generate(
                **self.mt_tokenizer_de.prepare_seq2seq_batch(list_src, return_tensors="pt").to(self.torch_device))

            # convert generated token indices into text
            translated_texts = [self.mt_tokenizer_de.decode(t, skip_special_tokens=True) for t in translated]

        if language == 'es':
            translated = self.mt_model_es.generate(
                **self.mt_tokenizer_es.prepare_seq2seq_batch(list_src, return_tensors="pt").to(self.torch_device))

            # convert generated token indices into text
            translated_texts = [self.mt_tokenizer_es.decode(t, skip_special_tokens=True) for t in translated]

        if language == 'ja':
            translated = self.mt_model_ja.generate(
                **self.mt_tokenizer_ja.prepare_seq2seq_batch(list_src, return_tensors="pt").to(self.torch_device))

            # convert generated token indices into text
            translated_texts = [self.mt_tokenizer_ja.decode(t, skip_special_tokens=True) for t in translated]

        return translated_texts

    def review_sentiment(self, text, language='en'):
        """
        Performs sentiment analysis on a review, for English by default
        :param text: text to analyse
        :param language: language for sentiment analyser, it's only English in expert.ai
        :return: a json object that includes overall sentiment and also phrase sentiment
        """
        output = None
        try:
            output = self.expertai_client.specific_resource_analysis(
             body={"document": {"text": text}}, params={'language': language, 'resource': 'sentiment'})
        except:
            pass
        return output

    def run_recursive(self, item):
        """
        Recursive procedure to extract all lemmata in phrases with sentiment
        :param item:
        :return: generator that includes lemmata and sentiment
        """
        yield {item.lemma: item.sentiment}
        if isinstance(item.items, list):
            for i in item.items:
                yield from self.run_recursive(i)

    def all_items_sentiment(self, input_data):
        """
        Extract fine-grained sentiment in the recursive fashion
        :return: a list of dictionaries for item sentiment
        """
        results = []

        for i in input_data:
            phrase = ''
            # extract lemmata recursively per phrase
            result = self.run_recursive(i)
            for r in result:
                for key in r:
                    phrase = phrase + " " + key
                    sentiment = r[key]
            # append all lemmata and the phrase sentiment score
            results.append({phrase : sentiment})

        return results

    def item_sentiment(self, input_data):
        """
        Extract fine-grained sentiment, limited to the pair of lemmata. Not currently used.
        :return: a list of dictionaries for item sentiment
        """
        results = []
        for i in input_data:
            if len(i.items) == 1:
                item = i.items[0].lemma + " " + i.lemma
                results.append({item: i.sentiment})
        return results

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
