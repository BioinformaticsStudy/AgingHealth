import pandas as pd

# reads files created by data_parser.py and split_data.py and creates a file containing information about that data
def readData(infiles,outfile):
    #columns to be output
    columns = ['rows', 'unq ids'] + ['wave ' + str(i) for i in range(10)] + ['unq wave ' + str(i) for i in range(10)]

    dataInfo = pd.DataFrame(index=[infiles[file] for file in infiles],columns=columns)
    for file in infiles:
        data = pd.read_csv(file)
        row = infiles[file]
        ids = data['id']
        dataInfo['rows'][row] = len(ids)
        dataInfo['unq ids'][row] = len(ids.unique())

        for i in range(10):
            thisWave = data.loc[data['wave'] == i]
            dataInfo['wave ' + str(i)][row] = len(thisWave)
            dataInfo['unq wave ' + str(i)][row] = len(thisWave['id'].unique())
        
    dataInfo.to_csv('../Data/data_info.csv')



fullData = "../Data/ELSA_cleaned.csv"
trainData = "../Data/train.csv"
testData = "../Data/test.csv"
validData = "../Data/valid.csv"

infiles = {fullData:"full_data", trainData:"train_data", testData:"test_data", validData:"valid_data"}

readData(infiles,"data_info")