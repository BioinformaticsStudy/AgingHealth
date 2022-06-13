from curses import def_shell_mode
import pandas as pd
import argparse
import random

parser = argparse.ArgumentParser('Sample')
parser.add_argument('--size', type=int, default=30,help='the amount of ids to be included in the sample')
args = parser.parse_args()
size = args.size

fullData = pd.read_csv('../Data/ELSA_cleaned.csv')
sampleData = fullData.drop_duplicates(subset=['id']).sample(size)
output = pd.DataFrame(columns=fullData.columns)
for i,row in sampleData.iterrows():
    id = row['id']
    individualData = fullData.loc[fullData['id']==id]
    output = output.append(individualData)

output.to_csv('../Data/sample_data.csv',index=False)

