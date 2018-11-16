# XML-CNN
Pytorch implementation of the paper http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf

## Dependencies

* NLTK (stopwords)
* Pytorch >= 0.3.1
* Gensim
* Matplotlib


Store Glove embeddings in the same directory as pre-trained embeddings. Download from [here](https://nlp.stanford.edu/data/glove.6B.zip) default embedding dimension is 300 with 6 Billion (6B) tokens. Otherwise you can set --model_variation = 0 for starting from scratch.

