# amazon-review-sentiment

Leveraging expert.ai functionality to analyse multilingual product reviews from Amazon.

To start indexing data with sentiment information follow the next steps:

1. To install requirements, run `pip install -r requirement.txt`

2. Replace credentials for expert.ai API in `review_sentiment/indexing.py`

3. Update `config.yaml` with your elastic configuration and the path to the review data, or any other text data you'd like to analyse.

4. To install Elasticsearch, pull the latest image and launch it in the terminal as follows (replace the version if needed):

`docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.11.1`

Once this is done, run `setup.py`.