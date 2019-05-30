import os, string
import numpy as np
from data_helpers import clean_str


def load_data_and_labels_from_files(datapath):
    """
    Loads hotel review data from files and assigns class labels.
    Returns data and labels.
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
    truthful_labels = [[0, 1] for _ in truthful_examples]
    deceptive_labels = [[1, 0] for _ in deceptive_examples]
    y = np.concatenate([truthful_labels, deceptive_labels], 0)
    return [x_text, y]


