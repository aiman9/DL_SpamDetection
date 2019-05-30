import os, string
import numpy as np

def load_embeddings_and_labels():
    """
        Load the BERT embeddings (already generated and saved) for the dataset, and assign labels.
        Returns embedding matrix and labels.
    """
    #change path to saved embedding file
    emb=np.load('./bert_embed/arr_0.npy')
    emb=emb.reshape(1600,2)

    # Generate labels
    deceptive_labels = [[1, 0] for _ in emb[:800]]
    truthful_labels = [[0, 1] for _ in emb[800:]]
    
    y = np.concatenate([deceptive_labels, truthful_labels], 0)
    return [emb[:,1], y]

def pad_and_reshape(emb_only, max_doc_length=864):

    """
        Pad all embeddings to equal length to generate input matrix for convolution.
        Returns padded embedding matrix.
    """
    blank_embed=np.zeros(768)

    for i in range(len(emb_only)):
        j=len(emb_only[i])
        while j<max_doc_length:
            emb_only[i].append(blank_embed)
            j+=1
        # if len(emb_only[i])!=max_doc_length:
        #     print(i,len(emb_only[i]))
        emb_only[i]=np.asarray(emb_only[i])

    return np.stack(emb_only)


