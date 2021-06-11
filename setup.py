import yaml
import os
import pandas as pd
from review_sentiment.indexing import Indexer
import time
import logging

logger = logging.getLogger(__name__)

# folder to load config file
CONFIG_PATH = "."


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config_dict = yaml.safe_load(file)
    return config_dict


if __name__ == '__main__':
    # load config
    config = load_config("config.yaml")
    logger.info(f"Loading configuration")
    for key, value in config.items():
        logger.info(f"{key} : {value}")

    if not os.path.exists(os.path.dirname(config["retriever"]["results"]["path"])):
        os.makedirs(os.path.dirname(config["retriever"]["results"]["path"]))

    host = config["indexer"]["host"]
    port = config["indexer"]["port"]
    username = config["indexer"]["username"]
    password = config["indexer"]["password"]
    index = config["indexer"]["index"]
    language = config["indexer"]["language"]
    retrieval_type = config["retriever"]["type"]
    top_k = config["retriever"]["top_k"]
    agg_fields = config["retriever"]["batch_field"]
    index = config["indexer"]["index"]
    synonym_field = config["indexer"]["synonym_field"]
    review_path = config["data"]["review_path"]

    # queries_file = config["data"]["query_path"]
    doc_path = config["data"]["doc_path"]
    metadata_path = config["data"]["metadata_path"]
    length_threshold = config["preprocess"]["length_threshold"]

    indexer = Indexer(host, port, username, password, index, language)

    # indexer.preprocess_reviews(review_path)
    # check if index exists; if not, create a new index
    # delete index, if need be
    indexer.delete_index()

    if (not indexer.document_store.client.indices.exists(config["indexer"]["index"])) \
            or (not indexer.document_store.get_all_documents(config["indexer"]["index"])):
        logger.info(f"Index is either empty or does not exist. Start indexing data")
        indexer.index_data(review_path)
    else:
        logger.info(f"Index exists and is non empty")
        logger.info(f"Index has the following stats: {indexer.data_stats()}")
