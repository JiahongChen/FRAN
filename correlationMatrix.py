import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
from os import listdir
from os.path import isfile, join

def get_df(algo):
    datasets = ['DE007', 'DE014', 'DE021', 'FE007', 'FE014', 'FE021', 'DE', 'FE']
    df = pd.DataFrame(columns = datasets)
    mypath = './record/'+algo
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for source in datasets:
        source_to_target_array = []
        for target in datasets:
            name = source+'-'+target
            for f in files:
                if '_'+name+'_' in f:
                    splited_file_name = f.split('_')
                    value = splited_file_name[2]
                    source_to_target_array.append(float(value)*100)
                    break
        df_row = pd.DataFrame([source_to_target_array], columns = datasets, index = [source])
        df = pd.concat([df, df_row])
    return df
def plot_corMatrix(df):
    print (df)
    print(np.mean(np.mean(df)))
    plt.rcParams['font.size'] = 14
    fig, (ax) = plt.subplots(1,1,figsize=(7.5,6))
    hm = sns.heatmap(df,
                    ax = ax,
                    cmap = 'coolwarm',
                    square=True,
                    annot=True,
                    fmt='.1f',
                    annot_kws={'size':14},
                    linewidths=0.05,
                    robust=True,
                    vmin = 50,
                    vmax = 100
                    )

    fig.subplots_adjust(top=0.93)
    plt.tight_layout()
    plt.savefig('FRAN_ave.pdf')


num_trials = 10
df = pd.DataFrame()
for i in range(1, num_trials+1):
    if i ==1:
        df = get_df('FRAN' +str(i)) 
    else:
        df += get_df('FRAN' +str(i)) 

df/=num_trials


plot_corMatrix(df)
