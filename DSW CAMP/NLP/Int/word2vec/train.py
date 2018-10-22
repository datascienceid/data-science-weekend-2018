import multiprocessing
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec

wiki = WikiCorpus('idwiki-latest-pages-articles.xml.bz2', 
                  lemmatize=False, dictionary={})
sentences = list(wiki.get_texts())
params = {'size': 500, 'window': 10, 'min_count': 10, 
          'workers': max(1, multiprocessing.cpu_count() - 1), 'sample': 1E-3,}
word2vec = Word2Vec(sentences, **params)
word2vec.save('idwiki500')
