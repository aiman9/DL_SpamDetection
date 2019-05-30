import os, string
import numpy as np
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"'", " ' ", string)
    # string = re.sub(r".", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels_from_files(datapath):
    """
    Loads files from 'datapath'.
    Returns list of documents and labels.
    """
    truthful_examples=[]
    deceptive_examples=[]
    for home, dirs, files in os.walk(datapath):
        for filename in files:
            if("truthful" in home):
                truthful_examples.append(open(home+'/'+filename).read())
            else:
                deceptive_examples.append(open(home+'/'+filename).read())

    truthful_examples = [s.strip() for s in truthful_examples]
    deceptive_examples = [s.strip() for s in deceptive_examples]
    
    # Split by words
    x_text = truthful_examples + deceptive_examples
    x_text = [clean_str(sent) for sent in x_text]
    
    # Generate labels
    truthful_labels = [0 for _ in truthful_examples]
    deceptive_labels = [1 for _ in deceptive_examples]
    y = np.concatenate([truthful_labels, deceptive_labels], 0)
    return [x_text, y]

