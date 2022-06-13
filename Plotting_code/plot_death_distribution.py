import pandas as pd
import seaborn as sns
import argparse
import numpy as np

parser = argparse.ArgumentParser('Plot_Death_Distribution')
parser.add_argument('--dataset', type=str,default='elsa', choices=['elsa','sample'],help='what dataset is read in; either \'elsa\' or \'sample\'')
parser.add_argument('--binsize',type=int,default=10)
args = parser.parse_args()
postfix, file = ('_sample', 'sample_data') if args.dataset=='sample' else ('','ELSA_cleaned')

data = pd.read_csv(f'../Data/{file}.csv')
deaths = data.groupby('id')['death age'].transform('first')
deaths = pd.DataFrame(deaths)
deaths = deaths.loc[deaths['death age'] > 0]

print(len(deaths))
bins = np.arange(20,120,args.binsize)
plot = sns.histplot(data=deaths, x='death age', bins=bins).set_title('Death Age Distribution')
fig = plot.get_figure()
fig.savefig(f'../Plots/death_distribution{postfix}.pdf')
