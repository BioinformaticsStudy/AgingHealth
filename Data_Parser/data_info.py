import pandas as pd
import argparse
import numpy as np
import seaborn as sns
# reads files created by data_parser.py and split_data.py and creates a file containing information about that data

parser = argparse.ArgumentParser('data info')
parser.add_argument('--dataset', type=str,default='elsa', choices=['elsa','sample'],help='what dataset is read in; either \'elsa\' or \'sample\'')
args = parser.parse_args()

def readData(infiles):
    #columns to be output
    columns = ['rows', 'unq ids'] + \
              ['wave ' + str(i) for i in range(10)] + \
              ['age <40s'] + [f'age {decade}s' for decade in range(40,110,10)] + ['age 110+'] + \
              ['start age <40s'] + [f'start age {decade}s' for decade in range(40,100,10)]

    dataInfo = pd.DataFrame(index=[infiles[file] for file in infiles],columns=columns).fillna(0)
    for file in infiles:
        data = pd.read_csv(file)
        row = infiles[file]
        ids = data['id']
        dataInfo['rows'][row] = len(ids)
        dataInfo['unq ids'][row] = len(ids.unique())
        
        # print(len(data.loc[data['age'].isnull]))
        for i in range(10):
            thisWave = data.loc[data['wave'] == i]
            dataInfo['wave ' + str(i)][row] = len(thisWave)
        dataInfo['age <40s'] = len(data.loc[data['age'] < 40])
        dataInfo['age 110+'] = len(data.loc[data['age'] >= 110])
        for decade in range(40,110,10):
            thisDecade = data.loc[(data['age'] >= decade) & (data['age'] < decade+10)]
            dataInfo[f'age {decade}s'][row] = len(thisDecade)
        for label,group in data.groupby('id'):
            decade = group['age'].min() // 10
            if decade < 4: 
                decade = '<4'
            dataInfo[f'start age {decade}0s'][row] += 1

    #calculate the difference between the full data and the sum of the other three rows
    differences = pd.DataFrame(index=['difference'],columns=dataInfo.columns)
    for i in range(len(dataInfo.columns)):
        col = dataInfo.columns[i]
        differences[col]['difference'] = dataInfo[col]['full_data']  - \
                                         dataInfo[col]['train_data'] - \
                                         dataInfo[col]['test_data']  - \
                                         dataInfo[col]['valid_data']
    dataInfo = dataInfo.append(differences)
    return dataInfo
    


postfix = '' if args.dataset=='elsa' else '_sample'

fullData = '../Data/sample_data.csv' if args.dataset=='sample' else "../Data/ELSA_cleaned.csv"
trainData = f"../Data/train{postfix}.csv"
testData = f"../Data/test{postfix}.csv"
validData = f"../Data/valid{postfix}.csv"

#map the file name to the row name
infiles = {fullData:"full_data", trainData:"train_data", testData:"test_data", validData:"valid_data"}

data = readData(infiles)
data.to_csv(f"../Data/data_info{postfix}.csv")
