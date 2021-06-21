# Multilingual sentiment analyser

Leveraging expert.ai functionality to analyse multilingual product reviews from Amazon.

The inspiration for this project stems from the fact that many organisations, from SMEs to corporates, gather user feedback yet are not always capable of analysing it in depth. It becomes even more challenging for organisations operating in different geographies as user feedback may include multilingual content and may be mixed in nature, e.g. users may provide positive feedback for some aspects of a product or service and criticise others in the same review. Sentiment analysis is often done at a coarse level (either positive or negative) and having fine-grained information on what customers may dislike would help organisations provide better services and products.

By leveraging expert.ai functionality, together with the state of the art tooling for information retrieval and machine translation, we make sure that user feedback collections

* are made searchable using review content
* can be filtered on different entries from the knowledge graph and sentiment as provided by expert.ai and other metadata
* are analysed at a fine-grained level by presenting phrases that indicate positive or negative sentiment to an end-user for a better understanding of mixed reviews
* are translated into English before they are processed by expert.ai sentiment module. This has been done because expert.ai offers sentiment analysis for English.

To exemplify diverse functionality of expert.ai we opted for the dataset of multilingual Amazon reviews on products as described here, in particular its subset on kitchen products. Every review is translated into English if its original language is not English (i.e., German, Japanese, or Spanish) and processed by expert.ai API to extract relevant entities and sentiment information. This utilises not only sentiment analysis outcomes but also information from the knowledge graph, which could be particularly useful for filtering reviews in the future. We also recursively extract all phrases that are judged as positive or negative, to provide more insights into product or service aspects that are valued or criticised by customers. The next step is to index extracted data into a search engine, whereby the original review is being augmented with the analysis from expert.ai. By doing so, organisations are able to analyse reviews at a more fine-grained level and go beyond standard review indexing. For information retrieval, we use bm25 that has been the standard in the industry for many years, though it can be easily replaced with other retrieval methods, such as dense retrieval techniques.

Our tech stack includes the back-end in python, front-end in React, and deployment via Netlify. For search functionality we use Elastic, for machine translation MarianMT models and expert.ai for linguistic and sentiment analysis. Elastic is being hosted in the AWS cloud environment. Since we are using Elastic stack for search, we have also built a Kibana dashboard, to exemplify the types of analyses that can be conducted on the indexed data, see our video. We have noticed that the number of review stars strongly correlates with the overall sentiment regardless of a language, which provides additional support for the use of machine-translated data.

# Installing

To start indexing data with sentiment information follow the next steps:

1. To install requirements, run `pip install -r requirement.txt`

2. Replace credentials for expert.ai API in `review_sentiment/indexing.py`

3. Update `config.yaml` with your elastic configuration and the path to the review data, or any other text data you'd like to analyse.

4. To install Elasticsearch, pull the latest image and launch it in the terminal as follows (replace the version if needed):

`docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.11.1`

Once this is done, run `setup.py`.