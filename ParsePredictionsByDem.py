import sys
import os
import numpy as np
import math
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, f1_score
import pandas as pd
import re
import ast


task = "Regression"#"Classification"
IS_REGRESSION = True
CSV_FILE = 'REG_DS4UD_5_Outcomes_SingleControlAgeFemale.csv'#'REG_CTLB_1grams_SingleControls.csv'

BOOTSTRAP_COUNT = 200
NUM_OF_CONTROLS=1
NUM_OF_OUTCOMES=5


correlationsDict = {}
errorCorrelationsDict = {}
p_yPredsDict = defaultdict(dict)
p_yTruesDict = defaultdict(dict)
p_terciles = defaultdict(dict)
pValueDict = []
fig, axes = plt.subplots(NUM_OF_CONTROLS * NUM_OF_OUTCOMES, 6, figsize=(40, 60))




def main():
    dfAllRuns = readDataFromCSV(CSV_FILE)
    results_df = iterateOverData(dfAllRuns)
    results_df.to_csv('results.csv')
    


def iterateOverData(dfAllRuns):

    approachesToRun = [
        ("Regression Test", "language"),
        ("Regression Test", "demographic"),
        ("Regression Test", "demographic_and_language"),
        ("Residualized Controls Regression Test", "demographic_and_language"),
        ("Factor Adaptation Regression Test", "demographic_and_language"),
        ("Residualized Factor Adaptation Regression Test", "demographic_and_language")
    ]
    baseline = ("Regression Test", "language")


    # Loop over all control/outcome/approach combinations
    results = []
    for (control, outcome), trial_df in dfAllRuns.groupby(['control', 'outcome']):
        for approach in approachesToRun:
            pred_df = _loadApproachColumn(trial_df, approach[0], approach[1], 'pred')
            true_df = _loadApproachColumn(trial_df, approach[0], 'true', 'true')
            cont_df = _loadApproachColumn(trial_df, approach[0], 'control_val', 'demographic_val')
            base_df = _loadApproachColumn(trial_df, baseline[0], baseline[1], 'base')

            cont_terciles_df = labelTerciles(cont_df, control)
            combined_df = pd.merge(pd.merge(pd.merge(pred_df, base_df, on='Id', how='inner'), cont_terciles_df[['Id', 'tercile']], on='Id', how='inner'), true_df, on='Id', how='inner').dropna()

            inverseParityRatio = calcMetricOnTerciles(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["tercile"]), disp_metric=lambda x: 1-minMaxRatio(x))
            inverseParityRatio_p = bootstrap(combined_df, disp_metric=lambda x: 1-minMaxRatio(x))
            correlation = calculateCorrelation(combined_df[["true", "pred"]])
            tercileCorrelations = calcMetricOnTerciles(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["tercile"]), disp_metric=lambda x: x)
            result = {
                'control': control,
                'outcome': outcome,
                'approach': approach,
                'tercileCorrelations' : tercileCorrelations,
                'inverseParity':inverseParityRatio,
                'inverseParityP': inverseParityRatio_p
            }
            results.append({**result, **correlation})

            print(approach)

    results_df = pd.DataFrame(results)
    results_df = results_df.pivot_table(
        index=['control', 'outcome'],
        columns=['approach'],
        values=None,
        aggfunc='first'
    )
    #print("COLS: ", results_df.columns)
    #results_df = pd.Categorical(results_df.columns[1], categories=approachesToRun, ordered=True)

    return results_df

def _loadApproachColumn(df, approach_name, approach_column, new_col_name):
    subset = df[df["test"] == approach_name][['Id', approach_column]].apply(pd.to_numeric, errors='coerce')
    subset.columns = ['Id', new_col_name]
    return subset


def readDataFromCSV(csv_file):
    
    dfs = defaultdict(dict)
    with open(csv_file, 'r') as file:
        lines = file.readlines()

    # Initialize variables
    current_approach = None
    current_approach_outcomes = None
    current_approach_demographic = None
    current_col_headers = None
    data = []

    #Read csv by line
    for i, line in enumerate(lines):

        # Read header
        if not line[0].isdigit():
            
            #Read first line (test information)
            if not current_approach:
                pattern = r"\[.*?\]|\b[^,\[]+\b"
                info_line = [
                    ast.literal_eval(match.strip()) if match.strip().startswith('[') else match.strip()
                    for match in re.findall(pattern, line)
                ]
                current_approach = info_line[0]
                current_approach_outcomes = [outcome for outcome in info_line[1] if outcome not in info_line[2]]
                current_approach_demographic = info_line[2]
                
            #Read second line (column headers)
            else:
                current_col_headers = line.strip().split(',')


        # Read data
        elif line.strip() and line[0].isdigit():
            dataLine = line.strip().split(',')
            data.append(dataLine)


        #Save section
        if i + 1 == len(lines) or (data and not lines[i + 1][0].isdigit()):
            df = pd.DataFrame(data, columns=current_col_headers)

            # Rename columns
            dfs_by_outcome = {}
            for outcome in current_approach_outcomes:
                columns = ['Id'] + [col for col in df.columns if(col.startswith(outcome + '_') or 'control' in col)]
                rename_dict = {
                    lambda col: "control" in col: "control_val",
                    lambda col: "__withLanguage" in col and ("_" + current_approach_demographic[0]) not in col: "language",
                    lambda col: "trues" in col: "true",
                    lambda col: "_" + current_approach_demographic[0] in col and "withLanguage" not in col and "control" not in col: "demographic",
                    lambda col: "_" + current_approach_demographic[0] + "_" in col and "withLanguage" in col: "demographic_and_language"
                }
                outcomeDf = df[columns]
                outcomeDf.columns = [
                    next((v for k, v in rename_dict.items() if k(col)), col)
                    for col in outcomeDf.columns
                ]
                #print("COLS: ", outcomeDf.columns)
                dfs_by_outcome[outcome] = outcomeDf
            
            #Save section data to dfs
            dfs[current_approach][current_approach_demographic[0]] = dfs_by_outcome
            data = []
            current_approach = None
            current_col_headers = None


    #Flatten the dictionary structure into a dataframe
    flattened_df = pd.concat(
        {approach: pd.concat(
            {demographic: pd.concat(outcome_data, names=['outcome'])
            for demographic, outcome_data in demographic_data.items()},
            names=['control']
        ) for approach, demographic_data in dfs.items()},
        names=['test']
    ).reset_index()
    flattened_df = _correctResiduals(flattened_df)

    return flattened_df



def _correctResiduals(df):
    for index, row in df[df['test'].str.contains('Residual', na=False)].iterrows():
        #print(float(df.at[index, 'demographic']))
        if(df.at[index, 'demographic']):
            df.at[index, 'demographic_and_language'] = float(df.at[index, 'demographic_and_language']) + float(df.at[index, 'demographic'])
            df.at[index, 'true'] = float(df.at[index, 'true']) + float(df.at[index, 'demographic'])
    return df



def calculateCorrelation(df):
    yPred = df['pred']
    yTrue = df['true']

    mse = mean_squared_error(yTrue, yPred)
    r_squared = r2_score(yTrue, yPred)
    pearson_corr, pearson_p = pearsonr(yTrue, yPred)
    spearman_corr, spearman_p = spearmanr(yTrue, yPred)

    results = {
        'length': len(yTrue),
        'mse': mse,
        'rSquared': r_squared,
        'pearson': pearson_corr,
        'pearsonP': pearson_p,
        'spearman': spearman_corr,
        'spearmanP': spearman_p
    }
    return results



'''
def discrete_gini_coefficient(pearson_r, length):
    # Mean of the weighted values
    mean_value = np.average(pearson_r, weights=length)
    
    # Gini coefficient calculation
    n = len(pearson_r)
    total_weight = np.sum(length)
    
    gini_sum = 0
    for i in range(n):
        for j in range(n):
            gini_sum += length[i] * length[j] * abs(pearson_r[i] - pearson_r[j])
    
    gini = gini_sum / (2 * total_weight * np.sum(length) * mean_value)
    
    return gini



def calculateGini(df):
    # Grouping and calculating Gini coefficient
    result = []
    group_size = 3
    
    # Iterate over DataFrame in groups of 3 rows
    for start in range(0, len(df), group_size):

        end = start + group_size
        group = df.iloc[start:end]
        
        # Calculate the Gini coefficient for each subcolumn
        for col in group.columns.levels[1]:  # Loop over 'pearson_r' and 'length'

            if(IS_REGRESSION):
                pearson_values = group[('Pearson_r', col)].values
            else:
                pearson_values = group[('AUC', col)].values
                
            length_values = group[('Length', col)].values
            
            gini = discrete_gini_coefficient(pearson_values, length_values)
            new_col = ('Gini Coefficient', col)

            # Ensure that the new column is float and not a categorical index
            if new_col not in df.columns:
                df[new_col] = pd.Series(dtype='float')
            #print("INDEX: ", group)

            # Assign the Gini coefficient to the specified row (index 2)
            # Use loc since new_col is a MultiIndex (a tuple)
            #df[new_col] = pd.Series(dtype='float')
            first_row = group.iloc[0]
            first_index = first_row.name
            df.loc[(first_index[0], first_index[1], first_index[2]), new_col] = gini
            
    return df
'''




tercile_mask_functions = {
    'logincomeHC01_VC85ACS3yr': lambda df: pd.cut(df, bins=[df.min()-1, df.quantile(1/3), df.quantile(2/3), df.max()], labels=['Low', 'Medium', 'High']),
    'hsgrad': lambda df: pd.cut(df, bins=[df.min()-1, df.quantile(1/3), df.quantile(2/3), df.max()], labels=['Low', 'Medium', 'High']),
    'forgnborn': lambda df: pd.cut(df, bins=[df.min()-1, df.quantile(1/3), df.quantile(2/3), df.max()], labels=['Low', 'Medium', 'High']),
    'age': lambda df: pd.cut(df, bins=[df.min()-1, df.quantile(1/3), df.quantile(2/3), df.max()], labels=['Low', 'Medium', 'High']),
    'is_female': lambda df: df.map({1: 'Medium', 0: 'Low', None: 'High'}),
    'is_black': lambda df: df.map({1: 'Medium', 0: 'Low', None: 'High'}),
    'individual_income': lambda df: pd.cut(df, bins=[df.min()-1, df.quantile(1/3), df.quantile(2/3), df.max()], labels=['Low', 'Medium', 'High']),
}

def labelTerciles(df, control):
    for key in tercile_mask_functions:
        if key in control:
            masks = tercile_mask_functions[key](pd.to_numeric(df["demographic_val"], errors='coerce'))
            df['tercile'] = masks
            return df
    raise ValueError("No matching mask function found for control: {}".format(control))



def bootstrap(df, internal_metric = pearsonr, disp_metric=lambda x: 1-minMaxRatio(x), num_resamples = BOOTSTRAP_COUNT):
    ypreds = df["pred"]
    ytrues = df["true"]
    old_ypreds = df["base"]
    tercile_ids = df["tercile"]
    count_null_trials = 0
    disp_new = calcMetricOnTerciles(ypreds, ytrues, tercile_ids, internal_metric, disp_metric)
    for k in range(num_resamples):
        indices = np.random.choice(len(df), size=len(df), replace=True)

        bs_old_ypreds = old_ypreds.reset_index(drop=True).loc[indices].tolist() 
        bs_ytrues = ytrues.reset_index(drop=True).loc[indices].tolist() 
        bs_tercile_ids = tercile_ids.reset_index(drop=True).loc[indices].tolist() 
 
        disp_old = calcMetricOnTerciles(bs_old_ypreds, bs_ytrues, bs_tercile_ids, internal_metric, disp_metric)
        if disp_old <= disp_new:
            count_null_trials += 1

    return count_null_trials / num_resamples


def bootstrapNew(df, internal_metric = pearsonr, disp_metric=lambda x: 1-minMaxRatio(x), num_resamples = BOOTSTRAP_COUNT):
    ypreds = df["pred"]
    ytrues = df["true"]
    old_ypreds = df["base"]
    tercile_ids = df["tercile"]
    count_null_trials = 0
    disp_old = calcMetricOnTerciles(old_ypreds, ytrues, tercile_ids, internal_metric, disp_metric)
    for k in range(num_resamples):
        indices = np.random.choice(len(df), size=len(df), replace=True)

        bs_ypreds = ypreds.reset_index(drop=True).loc[indices].tolist() 
        bs_ytrues = ytrues.reset_index(drop=True).loc[indices].tolist() 
        bs_tercile_ids = tercile_ids.reset_index(drop=True).loc[indices].tolist() 
 
        disp_new = calcMetricOnTerciles(bs_ypreds, bs_ytrues, bs_tercile_ids, internal_metric, disp_metric)
        if disp_old <= disp_new:
            count_null_trials += 1

    return count_null_trials / num_resamples


def bootstrapBoth(df, internal_metric = pearsonr, disp_metric=lambda x: 1-minMaxRatio(x), num_resamples = BOOTSTRAP_COUNT):
    ypreds = df["pred"]
    ytrues = df["true"]
    old_ypreds = df["base"]
    tercile_ids = df["tercile"]
    count_null_trials = 0
    
    for k in range(num_resamples):
        indices = np.random.choice(len(df), size=len(df), replace=True)

        bs_old_ypreds = old_ypreds.reset_index(drop=True).loc[indices].tolist() 
        bs_new_ypreds = ypreds.reset_index(drop=True).loc[indices].tolist() 
        bs_ytrues = ytrues.reset_index(drop=True).loc[indices].tolist() 
        bs_tercile_ids = tercile_ids.reset_index(drop=True).loc[indices].tolist() 

        disp_new = calcMetricOnTerciles(bs_new_ypreds, bs_ytrues, bs_tercile_ids, internal_metric, disp_metric)
        disp_old = calcMetricOnTerciles(bs_old_ypreds, bs_ytrues, bs_tercile_ids, internal_metric, disp_metric)
        if disp_old <= disp_new:
            count_null_trials += 1

    return count_null_trials / num_resamples



def calcMetricOnTerciles(ypreds, ytrues, tercile_ids, internal_metric = pearsonr, disp_metric=lambda x: 1-minMaxRatio(x)):
    numpyArrays = [np.array(var) for var in (ypreds, ytrues, tercile_ids)]
    ypreds, ytrues, tercile_ids = numpyArrays
    terc_dict = {
        "High": 2,
        "Medium": 1,
        "Low": 0
    }

    #Split values by tercile
    terc_ypreds = [[], [], []]
    terc_ytrues = [[], [], []]
    
    for i in range(len(ypreds)):
        this_tercile = terc_dict[tercile_ids[i]]
        terc_ypreds[this_tercile].append(ypreds[i])
        terc_ytrues[this_tercile].append(ytrues[i])

    #calculate the metric score per tercile
    results = []
    for t in range(3):
        cor, _ = internal_metric(terc_ypreds[t], terc_ytrues[t])
        results.append(cor)

    return disp_metric(results)
        

def minMaxRatio(rs):
    return np.min(rs) / np.max(rs)



'''def calcError(yPred, yTrue, axes):
    difference = [a - b for a, b in zip(yPred, yTrue)]
    if axes:
        sns.scatterplot(x='Control Value', y='Error', edgecolor='none', data=difference, alpha=0.3, s=6, ax=axes[col, names.index(name)])
        sns.regplot(x='Control Value', y='Error', data=difference, lowess=True, 
            scatter=False, line_kws={'color': 'red', 'linewidth': 2}, ax=axes[col, names.index(name)])

        if name == "Regression Test":
            titleName = "Regression (Lang and Controls) Test"
        else:
            titleName = name
        
        axes[col, names.index(name)].set_title('Error predicting ' + yTrueCol[:-6].replace('_', ' ') + ' using \n' + titleName[:-5] + ' vs ' + controlNames[str(control)], 
                                        fontsize=14, fontweight='bold')
        axes[col, names.index(name)].set_ylabel('Error in predicting ' + yTrueCol[:-6].replace('_', ' '), fontsize=12, fontweight='bold')
        axes[col, names.index(name)].set_xlabel(controlNames[str(control)], fontsize=12, fontweight='bold')'''



    
    
    
    


if __name__ == "__main__":

    main()


