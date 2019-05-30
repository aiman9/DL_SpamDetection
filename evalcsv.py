import sys
import pandas as pd
import numpy as np

#csv_path points to the desired csv file of predictions
if(len(sys.argv)>1):
    csv_path=sys.argv[1]
else:
    csv_path='./spamPredictions.csv'

df=pd.read_csv(csv_path, index_col=False, header=None)

predicted=np.asarray(df.iloc[:,1])
truth=np.asarray(df.iloc[:,-1])

# print(predicted.shape)

correct=0
incorrect=0
for i in range(len(df)):
    if(predicted[i]==truth[i]):
        correct+=1
    else:
        incorrect+=1

print("Accuracy = {}".format(correct/(correct+incorrect)))

