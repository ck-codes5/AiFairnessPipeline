import sys
import os
import numpy as np
from scipy.stats import kstest, uniform
from scipy.stats import linregress
from scipy.stats import zscore
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mticker
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
import textwrap


plt.rcParams.update({
    'font.size': 17,  # Adjust overall font size
    'axes.titlesize': 19,  # Title size
    'axes.titleweight': 'bold',  # Bold title
    'axes.labelsize': 17,  # Axis labels size
    'axes.labelweight': 'bold',  # Bold axis labels
    'xtick.labelsize': 14,  # X-axis tick labels
    'ytick.labelsize': 14,  # Y-axis tick labels
    'legend.fontsize': 14,  # Legend font size
    'figure.titlesize': 20,  # Figure title size
    'axes.linewidth': 2 
})
# plt.rcParams.update({
#     'font.size': 12,  # Adjust overall font size
#     'axes.titlesize': 15,  # Title size
#     #'axes.titleweight': 'bold',  # Bold title
#     'axes.labelsize': 13,  # Axis labels size
#     #'axes.labelweight': 'bold',  # Bold axis labels
#     'xtick.labelsize': 10,  # X-axis tick labels
#     'ytick.labelsize': 10,  # Y-axis tick labels
#     'legend.fontsize': 10,  # Legend font size
#     'figure.titlesize': 20,  # Figure title size
#     'axes.linewidth': 1 
# })
CHART_TITLE_WIDTH = 32


task = "Regression"#"Classification"
IS_REGRESSION = True
#CSV_FILE = "Regression_DS4UD_robertaEmbs_ControlsTested1AtATime_Oct15th_24_PaperVersion.csv"#'REG_DS4UD_5_Outcomes_SingleControlAgeFemale.csv'#'REG_CTLB_1grams_SingleControls.csv'
CSV_FILE = "Regression_CTLB_1grams_ControlsTested1AtATime_Oct15th_24_PaperVersion.csv"
ZSCORE = True
BOOTSTRAP_COUNT = 1
NUM_OF_CONTROLS=1
NUM_OF_OUTCOMES=4#5

cleanNames = {
        ("Regression Test", "language"): "Language Only",
        ("Regression Test", "demographic"): "Control Only",
        ("Regression Test", "demographic_and_language"): "Language and Control",
        ("Residualized Controls Regression Test", "demographic_and_language"): "Residualized Controls",
        ("Factor Adaptation Regression Test", "demographic_and_language"): "Factor Adaptation",
        ("Residualized Factor Adaptation Regression Test", "demographic_and_language"): "Residualized Factor Adaptation",
        "logincomeHC01_VC85ACS3yr$10": "Income",
        "hsgradHC03_VC93ACS3yr$10": "HS Graduation",
        "forgnbornHC03_VC134ACS3yr$10": "Foreign Born",
        "heart_disease": "Heart Disease",
        "life_satisfaction": "Life Satisfaction",
        "perc_fair_poor_health": "Fair Health",
        "suicide": "Suicide"
    }

approachesToRun = [
        ("Regression Test", "language"),
        # ("Regression Test", "demographic"),
        # ("Regression Test", "demographic_and_language"),
        # ("Residualized Controls Regression Test", "demographic_and_language"),
        # ("Factor Adaptation Regression Test", "demographic_and_language"),
        ("Residualized Factor Adaptation Regression Test", "demographic_and_language")
    ]

baseline = ("Regression Test", "language")

correlationsDict = {}
errorCorrelationsDict = {}
p_yPredsDict = defaultdict(dict)
p_yTruesDict = defaultdict(dict)
p_terciles = defaultdict(dict)
pValueDict = []
fig, axes = plt.subplots(NUM_OF_CONTROLS * NUM_OF_OUTCOMES, 6, figsize=(40, 60))



def main():
    dfAllRuns = readDataFromCSV('../features/' + CSV_FILE)
    results_df = iterateOverData(dfAllRuns)   
    results_df.to_csv('../results/resultsFrom_' + CSV_FILE)



def iterateOverData(dfAllRuns):

    plot_data = []
    # Loop over all control/outcome/approach combinations
    results = []
    for (control, outcome), trial_df in dfAllRuns.groupby(['control', 'outcome']):
        for approach in approachesToRun:
            pred_df = _loadApproachColumn(trial_df, approach[0], approach[1], 'pred')
            true_df = _loadApproachColumn(trial_df, approach[0], 'true', 'true')
            cont_df = _loadApproachColumn(trial_df, approach[0], 'control_val', 'demographic_val')
            base_df = _loadApproachColumn(trial_df, baseline[0], baseline[1], 'base')
            
            cont_terciles_df = labelTerciles(cont_df, control)
            combined_df = pd.merge(pd.merge(pd.merge(pred_df, base_df, on='Id', how='inner'), cont_terciles_df[['Id', 'tercile', 'demographic_val']], on='Id', how='inner'), true_df, on='Id', how='inner').dropna()
            
            '''if ZSCORE:
                combined_df['pred'] = zscore(combined_df['pred'])
                combined_df['true'] = zscore(combined_df['true'])'''

            #This function gives warnings during classification bc doesnt work
            #inverseParityRatio = calcMetricOnTerciles(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: 1-minMaxRatio(x), tercile_ids=list(combined_df["tercile"]))
            #inverseParityRatio_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: 1-minMaxRatio(x), terciles = True, compareWithNull=True)
            
            #giniCoefficient = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), disp_metric=lambda x: gini_coefficient(x))
            #giniCoefficient_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: discreteGiniCoefficient(x), terciles = False, compareWithNull=True)
            
            concentrationCurveSum = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: npConcentrationCoefficient(x)[1])
            concentrationCurve_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: npConcentrationCoefficient(x)[1], terciles = False, compareWithNull=False)

            absolute_diff = [abs(a - b) for a, b in zip(list(combined_df["pred"]), list(combined_df["true"]))]
            sorted_values = [v for _, v in sorted(zip(list(combined_df["demographic_val"]), absolute_diff))]
            #cumulative_share_of_population = np.linspace(0, 1, len(combined_df)+1)
            ksStat, ksStat_p = kstest(sorted_values, uniform.cdf, args=(0, 1))

            correlation = calculateCorrelation(combined_df[["true", "pred"]])
            tercileCorrelations = calcMetricOnTerciles(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: x, tercile_ids=list(combined_df["tercile"]))
            result = {
                'control': control,
                'outcome': outcome,
                'approach': approach,
                'tercileCorrelations' : tercileCorrelations,
                #'inverseParity':inverseParityRatio,
                #'inverseParityP': inverseParityRatio_p,
                #'giniCoefficient':giniCoefficient,
                #'giniCoefficientP': giniCoefficient_p
                #'concentrationCurveAbs': concentrationCurveAbs,
                'concentrationCurveSum': concentrationCurveSum,
                'concentrationCurveP': concentrationCurve_p
                #'ksStat': ksStat,
                #'ksStatP': ksStat_p
            }
            results.append({**result, **correlation})
            plot_data.append((combined_df, control, outcome, approach))

            print(approach)
            print(control, outcome)
            
    makePlotGrid(plot_data)

    results_df = pd.DataFrame(results)
    results_df = results_df.pivot_table(
        index=['control', 'outcome'],
        columns=['approach'],
        values=None,
        aggfunc='first'
    )
    level1, level2 = results_df.columns.levels
    level2_sorted = sorted(level2, key=lambda x: approachesToRun.index(x) if x in approachesToRun else float('inf'))
    sorted_columns = pd.MultiIndex.from_product([level1, level2_sorted], names=results_df.columns.names)
    results_df = results_df.reindex(columns=sorted_columns)
    return results_df



def _loadApproachColumn(df, approach_name, approach_column, new_col_name):
    subset = df[df["test"] == approach_name][['Id', approach_column]].apply(pd.to_numeric, errors='coerce')
    subset.columns = ['Id', new_col_name]
    return subset


def round_to_2_sig_figs(x):
    """Rounds a number to 2 significant figures without scientific notation."""
    if x == 0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(x))))  # Order of magnitude
    factor = 10**(magnitude - 1)  # Scale factor for two significant digits
    rounded = round(x / factor, 1) * factor  # Round and rescale
    return "{:.0f}".format(rounded) if magnitude >= 2 else "{:.1f}".format(rounded)



def makePlotGrid(plot_data):
    plotTypes = 2
    
    # Determine grid size
    n_cols = len(approachesToRun) * plotTypes # Number of columns in the grid
    n_rows = math.ceil(len(plot_data) / n_cols) * plotTypes  # Rows needed based on total plots

    # Create the grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), constrained_layout=True)
    plt.subplots_adjust(wspace=0.2, hspace=.55)
    axes = axes.flatten()  # Flatten in case of 2D array for easier indexing

    # Plot all scatterplots
    for i, (combined_df, control, outcome, approach) in enumerate(plot_data):

        row_index = plotTypes * i
        palette = sns.color_palette("deep", n_colors=3)  # Get 3 distinct colors
        unique_terciles = sorted(combined_df["tercile"].unique())  # Ensure consistent ordering
        #plotConcentrationCurve(axes[plotTypes * i], combined_df, control, outcome, approach)
        plotScatterplot(axes[row_index], combined_df, control, outcome, approach, palette, unique_terciles)
        plotScatterplotPredVsTrues(axes[row_index + 1], combined_df, control, outcome, approach, palette, unique_terciles)

        # Compute x position for the middle of both plots
        mid_x = (axes[row_index].get_position().x0 + axes[row_index + 1].get_position().x1) / 2

        # Add a shared title above both plots
        if(i%2==1):
            fig.text(
                mid_x, 
                axes[row_index].get_position().y1 + 0.006,  
                "Predicting {} using Lang and {}".format(cleanNames[outcome], cleanNames[control]), 
                fontdict={'fontsize': 20, 'fontweight': 'bold'}, 
                ha="center"
            )
        else:
            fig.text(
                mid_x, 
                axes[row_index].get_position().y1 + 0.006,  
                "Predicting {} using Lang Alone".format(cleanNames[outcome]), 
                fontdict={'fontsize': 20, 'fontweight': 'bold'}, 
                ha="center"
            )

    # Hide unused axes
    #for j in range(len(plot_data), len(axes)):
    #    axes[j].axis('off')

    # Add a single legend for the whole figure
    #handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, title="Tercile", loc="upper right")

    # Add a title for the whole grid
    fig.savefig("ConcentrationCurve.png", dpi=300)


def plotScatterplot(ax, combined_df, control, outcome, approach, palette, unique_terciles):

    combined_df["error"] = abs(combined_df["pred"] - combined_df["true"])
    tercile_colors = {tercile: palette[i] for i, tercile in enumerate(unique_terciles)}

    # Compute percentiles for trimming
    x_min, x_max = combined_df["demographic_val"].quantile([0.025, 0.975])
    y_min, y_max = combined_df["error"].quantile([0.025, 0.975])
    
    # Filter the data
    filtered_df = combined_df[
        (combined_df["demographic_val"] >= x_min) & 
        (combined_df["demographic_val"] <= x_max) & 
        (combined_df["error"] >= y_min) & 
        (combined_df["error"] <= y_max)
    ]

    sns.scatterplot(
        x="demographic_val", 
        y="error", 
        hue="tercile", 
        data=filtered_df,
        ax=ax,
        palette=tercile_colors,
        alpha=.9,
        legend=False,
        s=18,
        edgecolor='none',
        hue_order=unique_terciles
    )

    sns.regplot(
        x="demographic_val", 
        y="error", 
        data=filtered_df, 
        ax=ax,
        scatter=False, 
        lowess=True,
        color='black',
        line_kws={'lw': 5}
    )
    #title = "Predicting " + cleanNames[outcome] + " Using " + cleanNames[approach] + " With Respect To " + cleanNames[control]
    title = "Prediction Error"
    wrapped_title = "\n".join(textwrap.wrap(title, width=CHART_TITLE_WIDTH))
    # Reduce the number of ticks to 3
    ax.set_title(wrapped_title)
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3))

    # Format tick labels to two decimal places
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: round_to_2_sig_figs(y)))
    ax.set_xlabel(cleanNames[control], labelpad=-1)
    ax.set_ylabel('Absolute Error', labelpad=-3)


def plotScatterplotPredVsTrues(ax, combined_df, control, outcome, approach, palette, unique_terciles):

    #combined_df["error"] = abs(combined_df["pred"] - combined_df["true"])
    
    # Compute percentiles for trimming
    x_min, x_max = combined_df["pred"].quantile([0.025, 0.975])
    y_min, y_max = combined_df["true"].quantile([0.025, 0.975])
    
    # Filter the data
    filtered_df = combined_df[
        (combined_df["pred"] >= x_min) & 
        (combined_df["pred"] <= x_max) & 
        (combined_df["true"] >= y_min) & 
        (combined_df["true"] <= y_max)
    ]

    tercile_colors = {tercile: palette[i] for i, tercile in enumerate(unique_terciles)}


    sns.scatterplot(
        x="pred", 
        y="true", 
        hue="tercile", 
        data=filtered_df,
        ax=ax,
        palette=tercile_colors,
        alpha=.9,
        legend=False,
        s=18,
        edgecolor='none',
        hue_order=unique_terciles
    )

    '''sns.regplot(
        x="true", 
        y="pred", 
        data=combined_df, 
        ax=ax,
        scatter=False, 
        lowess=True,
        color='black',
        line_kws={'lw': 2}
    )'''

    legend_patches = []

    for tercile, sub_df in filtered_df.groupby("tercile"):

        slope, intercept, r_value, p_value, std_err = linregress(sub_df["pred"], sub_df["true"])
        
        transparent_color = mcolors.to_rgba(tercile_colors[tercile], alpha=0.4)
        sns.regplot(
            x="pred", 
            y="true", 
            data=sub_df, 
            ax=ax,
            scatter=False,   # Hide points
            color=transparent_color,  # Use tercile-specific color
            line_kws={'lw': 5, 'alpha': 0.5},
            ci=None
        )

        # Create a legend entry with the tercile name and slope
        #legend_patches.append(mlines.Line2D([0], [0], color=tercile_colors[tercile], lw=2, label="{} β:{:.2f}".format(tercile[0], slope)))

    #ax.legend(handles=legend_patches, loc="upper left", handlelength=1, handleheight=0.5, frameon=True, framealpha=0.3)

    # Add custom legend for beta coefficients
    #ax.legend(handles=legend_patches, loc="upper left")


    title = "True vs Predicted"
    wrapped_title = "\n".join(textwrap.wrap(title, width=CHART_TITLE_WIDTH))
    ax.set_title(wrapped_title)
    # Reduce the number of ticks to 3
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3))

    # Format tick labels to two decimal places
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: round_to_2_sig_figs(y)))
    ax.set_xlabel('Predicted', labelpad=-1)
    ax.set_ylabel('True', labelpad=-3)


def plotConcentrationCurve(ax, combined_df, control, outcome, approach, absolute=False):
    # Sort the data
    #rs = np.sort(rs)
    ypreds = combined_df["pred"]
    ytrues = combined_df["true"]
    demographics = combined_df["demographic_val"]
    absolute_diff = [abs(a - b) for a, b in zip(ypreds, ytrues)]

    rs = [v for _, v in sorted(zip(demographics, absolute_diff))]
    n = len(rs)

    cumulative_share_of_population = np.linspace(0, 1, n+1)
    cumulative_share_of_income = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    absoluteDeviationArea, sumDeviationArea = npConcentrationCoefficient(rs)

    # Plot the Lorenz curve
    lorenz_label = 'Concentration Curve \n(CI: {:.3f})'.format(sumDeviationArea)
    ax.plot(cumulative_share_of_population, cumulative_share_of_income, label=lorenz_label, color='blue')
    ax.plot([0, 1], [0, 1], label='Line of Equality', color='red', linestyle='--')
    ax.fill_between(cumulative_share_of_population, cumulative_share_of_income, cumulative_share_of_population, color='blue', alpha=0.2)
    
    # title = "Concentration Curve of Error of {} vs {} (Summed Deviation Area: {:.3f} Absolute Deviation Area: {:.3f})".format(
    #     cleanNames[approach], cleanNames[control], sumDeviationArea, absoluteDeviationArea
    # )
    title = "Concentration Curve"
    wrapped_title = "\n".join(textwrap.wrap(title, width=CHART_TITLE_WIDTH))
    ax.set_title(wrapped_title)
    # Reduce the number of ticks to 3
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3))

    # Format tick labels to two decimal places
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: round_to_2_sig_figs(y)))
    ax.set_xlabel('Cumul. prop. of counties')
    ax.set_ylabel('Cumul pred. error', labelpad=-5)
    ax.legend()
    '''ax.title('Lorenz Curve (Area: ' + str(area_under_lorenz_curve) + ' Deviation: ' +  str(gini_coefficient) + ')')
    ax.xlabel('Cumulative Share of Population Sample: ' + str(rs[0]))
    plt.ylabel('Cumulative Share of Wealth/Income')
    plt.legend()
    plt.grid(True)
    
    # Save to file
    plt.savefig("../lorenz_curve.png", bbox_inches='tight')
    plt.close()'''



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

def gini_coefficient(rs):
    # Ensure incomes are sorted
    rs = np.sort(rs)
    # Number of incomes
    n = rs.size
    # Cumulative sum of incomes
    cumulative_rs = np.cumsum(rs, dtype=float)
    # Gini coefficient formula
    gini = (2 / n) * (np.sum((np.arange(1, n + 1) * rs)) / cumulative_rs[-1]) - (n + 1) / n
    return gini


def npConcentrationCoefficient(rs):
    # Sort the data
    #rs = np.sort(rs)
    n = len(rs)

    # Calculate cumulative proportions
    cumulative_share_of_population = np.linspace(0, 1, n+1)
    cumulative_share_of_income = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    # Compute the absolute deviation from the equality line
    absoluteDeviationCurve = np.abs(cumulative_share_of_income - cumulative_share_of_population)
    sumDeviationArea = 2 * ((np.trapz(cumulative_share_of_income, cumulative_share_of_population)) - .5)
    absoluteDeviationArea = 2 * np.trapz(absoluteDeviationCurve, cumulative_share_of_population)
    return absoluteDeviationArea, sumDeviationArea



def npGiniCoefficient(rs):
    # Sort the data
    rs = np.sort(rs)
    n = len(rs)
    # Calculate cumulative proportions
    cumulative_share_of_population = np.linspace(0, 1, n+1)
    cumulative_share_of_income = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    # Calculate the area under the Lorenz curve
    area_under_lorenz_curve = np.trapz(cumulative_share_of_income, cumulative_share_of_population)
    gini_coefficient = 1 - 2 * area_under_lorenz_curve  # Gini coefficient

    return gini_coefficient



def discrete_gini_coefficient_complicateed(pearson_r, length):
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



def discreteGiniCoefficient(rs):
    
    # Gini coefficient calculation
    rs = np.sort(rs)
    n = len(rs)
    cumulative_rs = np.cumsum(rs)
    #cumulative_rs - cumulative_rs[0]
    mean_value = np.average(cumulative_rs)
    #print(cumulative_rs)
    gini_sum = 0
    for i in range(n-1):
        gini_sum += (2*i-n-1) * (cumulative_rs[i])
        #print("I: ", gini_sum)
    
    #print("cumunfes: ", (.5 * cumulative_rs[-1]), " SUM: ", gini_sum)
    gini = 1- (gini_sum / (n*n*mean_value))
    
    return gini
    '''# Gini coefficient calculation
    rs = np.sort(rs)
    n = len(rs)
    #print(rs)
    cumulative_rs = np.cumsum(rs)
    cumulative_rs - cumulative_rs[0]
    #print(cumulative_rs)
    gini_sum = 0
    for i in range(n-1):
        gini_sum += .5 * (1/n) * (cumulative_rs[i] + cumulative_rs[i+1])
        #print("I: ", gini_sum)
    
    #print("cumunfes: ", (.5 * cumulative_rs[-1]), " SUM: ", gini_sum)
    gini = (.5 * cumulative_rs[-1]) - gini_sum
    
    return gini'''
    '''# Gini coefficient calculation
    n = len(rs)
    # Mean of the weighted values
    mean_value = np.average(rs)
    
    rs = [0] + rs
    gini_sum = 0
    for i in range(n):
        gini_sum += rs[i] + rs[i+1]
    
    gini = 1 - (gini_sum / (n * rs[-1]))
    
    return gini'''
    '''
    # Gini coefficient calculation
    n = len(rs)
    # Mean of the weighted values
    mean_value = np.average(rs)
    
    
    gini_sum = 0
    for i in range(n):
        gini_sum += (2 * i - n - 1) * rs[i]
    
    gini = gini_sum / (mean_value * n * n)
    
    return gini'''



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



def bootstrapResampleBase(df, internal_metric = pearsonr, disp_metric=lambda x: 1-minMaxRatio(x), num_resamples = BOOTSTRAP_COUNT, terciles =True, compareWithNull = False):
    ypreds = df["pred"]
    ytrues = df["true"]
    base_ypreds = df["base"]
    tercile_ids = df["tercile"]
    count_null_trials = 0

    if(terciles):
        calculateMetric = lambda *args, **kwargs: calcMetricOnTerciles(*args, **kwargs)
    else:
        calculateMetric = lambda *args, **kwargs: calcMetricOnFullData(*args, **kwargs)

    disp_new = calculateMetric(ypreds, ytrues, disp_metric, tercile_ids, internal_metric)
    for k in range(num_resamples):
        indices = np.random.choice(len(df), size=len(df), replace=True)

        bs_base_ypreds = base_ypreds.reset_index(drop=True).loc[indices].tolist() 
        bs_ytrues = ytrues.reset_index(drop=True).loc[indices].tolist() 
        bs_tercile_ids = tercile_ids.reset_index(drop=True).loc[indices].tolist() 
 
        disp_old = calculateMetric(bs_base_ypreds, bs_ytrues, disp_metric, bs_tercile_ids, internal_metric)
        if abs(disp_base) <= abs(disp_new):
            count_null_trials += 1

    #print("AARON_DEBUG_RATIO: ", count_null_trials / num_resamples)
    return count_null_trials / num_resamples



def bootstrapResampleAlternative(df, internal_metric = pearsonr, disp_metric=lambda x: 1-minMaxRatio(x), num_resamples = BOOTSTRAP_COUNT, terciles = True, compareWithNull = False):
    ypreds = df["pred"]
    ytrues = df["true"]
    base_ypreds = df["base"]
    tercile_ids = df["tercile"]
    count_null_trials = 0

    if(terciles):
        calculateMetric = lambda *args, **kwargs: calcMetricOnTerciles(*args, **kwargs)
    else:
        calculateMetric = lambda *args, **kwargs: calcMetricOnFullData(*args, **kwargs)

    if compareWithNull:
        tercile_ids = pd.Series(np.random.permutation(tercile_ids.values), index=tercile_ids.index)
        disp_base = calculateMetric(ypreds, ytrues, disp_metric, tercile_ids, internal_metric)
    else:
        disp_base = calculateMetric(base_ypreds, ytrues, disp_metric, tercile_ids, internal_metric)
    for k in range(num_resamples):
        indices = np.random.choice(len(df), size=len(df), replace=True)

        bs_ypreds = ypreds.reset_index(drop=True).loc[indices].tolist() 
        bs_ytrues = ytrues.reset_index(drop=True).loc[indices].tolist() 
        bs_tercile_ids = tercile_ids.reset_index(drop=True).loc[indices].tolist() 
 
        disp_new = calculateMetric(bs_ypreds, bs_ytrues, disp_metric, bs_tercile_ids, internal_metric)
        if disp_base <= disp_new:
            count_null_trials += 1

    return count_null_trials / num_resamples



def bootstrapResampleBoth(df, internal_metric = pearsonr, disp_metric=lambda x: 1-minMaxRatio(x), num_resamples = BOOTSTRAP_COUNT, terciles = True, compareWithNull = False):
    ypreds = df["pred"]
    ytrues = df["true"]
    base_ypreds = df["base"]
    tercile_ids = df["tercile"]
    demographics = df["demographic_val"]
    count_null_trials = 0

    if(terciles):
        calculateMetric = lambda *args, **kwargs: calcMetricOnTerciles(*args, **kwargs)
    else:
        calculateMetric = lambda *args, **kwargs: calcMetricOnFullData(*args, **kwargs)


    for k in range(num_resamples):
        
        indices_new = np.random.choice(len(df), size=len(df), replace=True)
        bs_new_ypreds = ypreds.reset_index(drop=True).loc[indices_new].tolist() 
        bs_new_ytrues = ytrues.reset_index(drop=True).loc[indices_new].tolist() 
        bs_new_tercile_ids = tercile_ids.reset_index(drop=True).loc[indices_new].tolist() 
        bs_new_demographics = demographics.reset_index(drop=True).loc[indices_new].tolist() 

        disp_new = calculateMetric(bs_new_ypreds, bs_new_ytrues, demographics=bs_new_demographics, disp_metric=disp_metric, tercile_ids=bs_new_tercile_ids, internal_metric = internal_metric)
        if compareWithNull:
            if(terciles):
                bs_new_tercile_ids = np.random.permutation(bs_new_tercile_ids)
                disp_base = calculateMetric(bs_new_ypreds, bs_new_ytrues, demographics=demographics, disp_metric=disp_metric, tercile_ids=bs_new_tercile_ids, internal_metric = internal_metric)
            else:
                indices_new = np.random.choice(len(df), size=len(df), replace=True)
                bs_new_ypreds = ypreds.reset_index(drop=True).loc[indices_new].tolist() 
                bs_new_ytrues = ytrues.reset_index(drop=True).loc[indices_new].tolist() 
                bs_new_tercile_ids = tercile_ids.reset_index(drop=True).loc[indices_new].tolist()
                disp_base = calculateMetric(bs_new_ypreds, bs_new_ytrues, demographics=demographics, disp_metric=disp_metric, tercile_ids=bs_new_tercile_ids, internal_metric = internal_metric)
        else:
            indices_base = np.random.choice(len(df), size=len(df), replace=True)
            bs_base_ypreds = base_ypreds.reset_index(drop=True).loc[indices_base].tolist() 
            bs_base_ytrues = ytrues.reset_index(drop=True).loc[indices_base].tolist() 
            bs_base_tercile_ids = tercile_ids.reset_index(drop=True).loc[indices_base].tolist() 
            bs_base_demographics = demographics.reset_index(drop=True).loc[indices_base].tolist() 
            disp_base = calculateMetric(bs_base_ypreds, bs_base_ytrues, demographics=bs_base_demographics, disp_metric=disp_metric, tercile_ids=bs_base_tercile_ids, internal_metric = internal_metric)
        #print("DISPARITIES: ", disp_new, disp_base)
        if abs(disp_base) <= abs(disp_new):
            count_null_trials += 1

    return count_null_trials / num_resamples



def calcMetricOnTerciles(ypreds, ytrues, demographics, disp_metric=lambda x: 1-minMaxRatio(x), tercile_ids=None, internal_metric = pearsonr):
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

    #print("Test: ", terc_ypreds[0][:5])

    #calculate the metric score per tercile
    results = []
    for t in range(3):
        cor, _ = internal_metric(terc_ypreds[t], terc_ytrues[t])
        results.append(cor)

    return disp_metric(results)



def calcMetricOnFullData(ypreds, ytrues, demographics=None, disp_metric=lambda x: calculateGini(x), tercile_ids=None, internal_metric = None):

    absolute_diff = [abs(a - b) for a, b in zip(ypreds, ytrues)]

    sorted_values = [v for _, v in sorted(zip(demographics, absolute_diff))]

    result = disp_metric(sorted_values)

    return result
        


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


