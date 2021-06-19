import yaml
import os
from review_sentiment.indexing import Indexer
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

    host = config["indexer"]["host"]
    port = config["indexer"]["port"]
    username = config["indexer"]["username"]
    password = config["indexer"]["password"]
    index = config["indexer"]["index"]
    language = config["indexer"]["language"]
    review_path = config["data"]["review_path"]

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
