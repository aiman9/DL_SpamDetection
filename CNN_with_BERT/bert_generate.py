import numpy as np
import os, string
from bert_embedding import BertEmbedding

def read_filenames(datapath):
    file_list=[]
    for home, dirs, files in os.walk(datapath):
        for filename in files:
            file_list.append(home+'/'+filename)
    return sorted(file_list)
    


def load_data_and_labels_from_files(files):    
    x_text=[]

    for filename in files:
        x_text.append(open(filename).read())

    x_text = [s.strip() for s in x_text]

    # Generate labels
    deceptive_labels = [[1, 0] for _ in x_text[:800]]
    truthful_labels = [[0, 1] for _ in x_text[800:]]
    
    y = np.concatenate([deceptive_labels, truthful_labels], 0)
    return [x_text, y]

def bertTransform(x):
    """
        Generates and returns BERT word embeddings for the document list x
    """
    bert_embedding = BertEmbedding(max_seq_length=900)
    for i in range(0,len(x),2):
        result = bert_embedding(x[i:i+2])
        yield result
        print("{}/1600 done".format(i+2))
    

def main(argv=None):
    #Read sorted filenames
    files=read_filenames('../Spam_Detection_Data/')
    print(files)

    #Load files from filename list
    print("Loading data...")
    x,y=load_data_and_labels_from_files(files)
    print(len(y))
            
    print("Data Loaded.\nCreating Bert Embeddings...")
    #Generate BERT embeddings and save them as numpy array
    z=np.asarray(list(bertTransform(x)))
    np.savez('bert_embed',z)


if __name__ == '__main__':
    main()