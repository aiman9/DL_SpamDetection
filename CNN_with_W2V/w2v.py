import gensim
import os
import string
import re
import numpy as np
import inputfiles
from tensorflow.contrib import learn

def vocabMaker(x_text):

    """
        Creates embedding matrix for the vocabulary of the dataset.
        Uses Google's Word2Vec pretrained model for embeddings.
        Model used can be downloaded from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    """
    print("Loading W2V model...")
    model=gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
    print("Loaded")

    # Build vocabulary
    max_document_length = max([len(d.split(" ")) for d in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    d=vocab_processor.vocabulary_._mapping

    wt=[]
    for i in d:
        try:
            wt.append(model[i])
        except Exception:
            wt.append(np.zeros(300))
    w=np.asarray(wt)
    np.savez('init_wt', t=w)
    # print("Shape of np array being saved:",w.shape)

    return x, wt, vocab_processor