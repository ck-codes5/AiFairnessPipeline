#Import Libraries
import sys
import os
import numpy as np
from scipy.integrate import quad
from scipy.stats import kstest, uniform
from scipy.stats import linregress
from scipy.stats import zscore
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mticker
from scipy.stats import norm
import math
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.stats import chi2_contingency
from collections import defaultdict
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, f1_score
import pandas as pd
import re
import ast
import textwrap



#Set the dimensions of the graphs that are created with matplotlib
plt.rcParams.update({
    'font.size': 17,  # Adjust overall font size
    'axes.titlesize': 19,  # Title size
    'axes.titleweight': 'bold',  # Bold title
    'axes.labelsize': 14,  # Axis labels size
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
fig, axes = plt.subplots(NUM_OF_CONTROLS * NUM_OF_OUTCOMES, 6, figsize=(40, 60))



#Global variables for testing different approaches (you shouldnt need to edit these)
task = "Regression"
IS_REGRESSION = True
#CSV_FILE = "Regression_DS4UD_robertaEmbs_ControlsTested1AtATime_Oct15th_24_PaperVersion.csv"#'REG_DS4UD_5_Outcomes_SingleControlAgeFemale.csv'#'REG_CTLB_1grams_SingleControls.csv'
CSV_FILE = "Regression_CTLB_1grams_ControlsTested1AtATime_Oct15th_24_PaperVersion.csv"
num_bins = 3
ZSCORE = False
BOOTSTRAP_COUNT = 1000
NUM_OF_CONTROLS=1
NUM_OF_OUTCOMES=4#5



#cleaned, shortened names for printing more legible names in results
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

#All the combinations of approaches to run
approachesToRun = [
        ("Regression Test", "language"),
        #("Regression Test", "demographic"),
        #("Regression Test", "demographic_and_language"),
        #("Residualized Controls Regression Test", "demographic_and_language"),
        ("Factor Adaptation Regression Test", "demographic_and_language"),
        #("Residualized Factor Adaptation Regression Test", "demographic_and_language")
    ]

#The baseline approach to compare to
baseline = ("Regression Test", "language")




def main():
    dfAllRuns = readDataFromCSV('../features/' + CSV_FILE)
    results_df = iterateOverData(dfAllRuns)   
    results_df.to_csv('../results/resultsFrom_' + CSV_FILE)



# Main loop that performs bias tests and correlations on the prediction data
def iterateOverData(dfAllRuns):

    plot_data = []
    results = []

    # Loop over all control/outcome/approach combinations
    for (control, outcome), trial_df in dfAllRuns.groupby(['control', 'outcome']):
        for approach in approachesToRun:

            #Get a dataframe with columns (pred, true, demographic_val, base) to perform calculations with
            pred_df = _loadApproachColumn(trial_df, approach[0], approach[1], 'pred')
            true_df = _loadApproachColumn(trial_df, approach[0], 'true', 'true')
            cont_df = _loadApproachColumn(trial_df, approach[0], 'control_val', 'demographic_val')
            base_df = _loadApproachColumn(trial_df, baseline[0], baseline[1], 'base')
            cont_bins_df = labelBins(cont_df, control)
            combined_df = pd.merge(pd.merge(pd.merge(pred_df, base_df, on='Id', how='inner'), cont_bins_df[['Id', 'bin', 'demographic_val']], on='Id', how='inner'), true_df, on='Id', how='inner').dropna()
            
            #Zscore the data if desired
            if ZSCORE:
                combined_df['pred'] = zscore(combined_df['pred'])
                combined_df['true'] = zscore(combined_df['true'])


            #inverseParityRatio = calcMetricOnBins(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: 1-minMaxRatio(x), bin_ids=list(combined_df["bin"]))
            #inverseParityRatio_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: 1-minMaxRatio(x), bins = True, compareWithNull=True)
            
            #giniCoefficient = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), disp_metric=lambda x: gini_coefficient(x))
            #giniCoefficient_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: discreteGiniCoefficient(x), bins = False, compareWithNull=True)
            
            #concentrationCurveSum = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: npConcentrationCoefficient(x)[0])
            #concentrationCurve_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: npConcentrationCoefficient(x)[0], bins = False, compareWithNull=False)

            absolute_diff = [abs(a - b) for a, b in zip(list(combined_df["pred"]), list(combined_df["true"]))]
            sorted_values = [v for _, v in sorted(zip(list(combined_df["demographic_val"]), absolute_diff))]

            #cumulative_share_of_population = np.linspace(0, 1, len(combined_df)+1)
            #ksStat, ksStat_p = kstest(sorted_values, uniform.cdf, args=(0, 1))

            #ksStat = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: KsTest(x))
            #ksStat_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: KsTest(x), bins = False, compareWithNull=True)
            
            #likelihood = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcLikelihood(x))
            #likelihood_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: calcLikelihood(x), bins = False, compareWithNull=False)

            #JensenShannon = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcJensenShannon(x))
            #JensenShannon_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: calcJensenShannon(x), bins = False, compareWithNull=False)
            
            #chiSquared = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcChiSquared(x))
            #chiSquared_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: calcChiSquared(x), bins = False, compareWithNull=False)
            #chiSquaredBin, chiSquaredBin_p = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcChiSquared(x), bin_ids=list(combined_df["bin"]))#, internal_metric=lambda x, y: np.mean(np.abs(np.array(x) - np.array(y))))
            #chiSquaredBin_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: 1-minMaxRatio(x), bins = True, compareWithNull=True)
            
            # ksTest = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: KsTest(x))
            # ksTestNull_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: KsTest(x), bins = False, compareWithNull=True)
            # ksTestOther_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: KsTest(x), bins = False, compareWithNull=False)
            
            #customConcentrationValue = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: npConcentrationCoefficient(x))
            # customConcentrationValueNull_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: npConcentrationCoefficient(x), bins = False, compareWithNull=True)
            # customConcentrationValueOther_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: npConcentrationCoefficient(x), bins = False, compareWithNull=False)
            #customConcentrationIntegrateValue = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: npConcentrationCoefficientIntegrate(x))
            
            # andersonDarling = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcAndersonDarling(x))
            # andersonDarlingNull_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: calcAndersonDarling(x), bins = False, compareWithNull=True)
            # andersonDarlingOther_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: calcAndersonDarling(x), bins = False, compareWithNull=False)
            
            # andysMetricBin = calcMetricOnBins(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcAndyDeviation(x), bin_ids=list(combined_df["bin"]), internal_metric=lambda x, y: np.mean(np.abs(np.array(x) - np.array(y))))
            # andysMetricBinNull_p = bootstrapResampleBoth(combined_df, internal_metric=lambda x, y: np.mean(np.abs(np.array(x) - np.array(y))), disp_metric=lambda x: calcAndyDeviation(x), bins = True, compareWithNull=True)
            # andysMetricBinOther_p = bootstrapResampleBoth(combined_df, internal_metric=lambda x, y: np.mean(np.abs(np.array(x) - np.array(y))), disp_metric=lambda x: calcAndyDeviation(x), bins = True, compareWithNull=False)

            # crossEntropyBin = calcMetricOnBins(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcCrossEntropy(x), bin_ids=list(combined_df["bin"]), internal_metric=lambda x, y: np.mean(np.abs(np.array(x) - np.array(y))))
            # crossEntropyBinNull_p = bootstrapResampleBoth(combined_df, internal_metric=lambda x, y: np.mean(np.abs(np.array(x) - np.array(y))), disp_metric=lambda x: calcCrossEntropy(x), bins = True, compareWithNull=True)
            # crossEntropyBinOther_p = bootstrapResampleBoth(combined_df, internal_metric=lambda x, y: np.mean(np.abs(np.array(x) - np.array(y))), disp_metric=lambda x: calcCrossEntropy(x), bins = True, compareWithNull=False)

            correlation = calculateCorrelation(combined_df[["true", "pred"]])
            binCorrelations = calcMetricOnBins(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: x, bin_ids=list(combined_df["bin"]))
            
            
            #Save results to a dictionary
            result = {
                'control': control,
                'outcome': outcome,
                'approach': approach
                #'binCorrelations' : binCorrelations,
                #'inverseParity':inverseParityRatio,
                #'inverseParityP': inverseParityRatio_p,
                #'giniCoefficient':giniCoefficient,
                #'giniCoefficientP': giniCoefficient_p
                #'concentrationCurveAbs': concentrationCurveAbs,
                #'concentrationCurveSum': concentrationCurveSum,
                #'concentrationCurveP': concentrationCurve_p
                #'ksStat': ksStat,
                #'ksStatP': ksStat_p
                #'likelihood': likelihood,
                #'likelihoodP': likelihood_p
                #'JensenShannon': JensenShannon,
                #'JensenShannonP': JensenShannon_p,
                #'chiSquared': chiSquaredBin,
                #'chiSquaredBinP': chiSquaredBin_p,
                # 'andersonDarling': andersonDarling,
                # 'andersonDarlingNullP': andersonDarlingNull_p,
                # 'andersonDarlingOtherP': andersonDarlingOther_p,
                # 'andysMetricBin': andysMetricBin,
                # 'andysMetricBinOtherP': andysMetricBinOther_p,
                # 'andysMetricBinNullP': andysMetricBinNull_p
                # 'ksTest': ksTest,
                # 'ksTestNullP': ksTestNull_p,
                # 'ksTestOtherP': ksTestOther_p,
                #'customConcentrationValue': customConcentrationValue,
                #'customConcentrationIntegrateValue': customConcentrationIntegrateValue
                # 'customConcentrationValueNullP': customConcentrationValueNull_p,
                # 'customConcentrationValueOtherP': customConcentrationValueOther_p
                #'crossEntropyValue': crossEntropyBin,
                #'crossEntropyValueNullP': crossEntropyBinNull_p,
                #'crossEntropyValueOtherP': crossEntropyBinOther_p
            }
            results.append({**result, **correlation})
            plot_data.append((combined_df, control, outcome, approach))

            print(approach)
            print(control, outcome)
            
    #Run code to generate plots
    makePlotGrid(plot_data)

    #Save all those results dictionaries we created into a pandas dataframe
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



#load the column for a specific approach from the total dataframe
def _loadApproachColumn(df, approach_name, approach_column, new_col_name):
    subset = df[df["test"] == approach_name][['Id', approach_column]].apply(pd.to_numeric, errors='coerce')
    subset.columns = ['Id', new_col_name]
    return subset



#Rounds a number to 2 significant figures without scientific notation.
def round_to_2_sig_figs(x):
    if x == 0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(x))))  # Order of magnitude
    factor = 10**(magnitude - 1)  # Scale factor for two significant digits
    rounded = round(x / factor, 1) * factor  # Round and rescale
    return "{:.0f}".format(rounded) if magnitude >= 2 else "{:.1f}".format(rounded)



#Make grid of plots based on prediction data
def makePlotGrid(plot_data):
    plotTypes = 1
    
    # Determine grid size
    n_cols = len(approachesToRun) * plotTypes # Number of columns in the grid
    n_rows = math.ceil(len(plot_data) / n_cols) * plotTypes  # Rows needed based on total plots

    # Create the grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), constrained_layout=True)
    plt.subplots_adjust(wspace=0.2, hspace=.55)
    axes = axes.flatten()


    # Plot all scatterplots
    for i, (combined_df, control, outcome, approach) in enumerate(plot_data):

        row_index = plotTypes * i
        palette = sns.color_palette("deep", n_colors=3)  # Get 3 distinct colors
        unique_bins = sorted(combined_df["bin"].unique())  # Ensure consistent ordering
        plotConcentrationCurve(axes[plotTypes * i], combined_df, control, outcome, approach)
        #plotKSCurve(axes[plotTypes * i], combined_df, control, outcome, approach)
        #plotScatterplot(axes[row_index], combined_df, control, outcome, approach, palette, unique_bins)
        #plotScatterplotPredVsTrues(axes[row_index + 1], combined_df, control, outcome, approach, palette, unique_bins)

        # Compute x position for the middle of both plots
        #mid_x = (axes[row_index].get_position().x0 + axes[row_index + 1].get_position().x1) / 2

        # Add a shared title above both plots
        # if(i%2==1):
        #     fig.text(
        #         mid_x, 
        #         axes[row_index].get_position().y1 + 0.006,  
        #         "Predicting {} using Lang and {}".format(cleanNames[outcome], cleanNames[control]), 
        #         fontdict={'fontsize': 20, 'fontweight': 'bold'}, 
        #         ha="center"
        #     )
        # else:
        #     fig.text(
        #         mid_x, 
        #         axes[row_index].get_position().y1 + 0.006,  
        #         "Predicting {} using Lang Alone".format(cleanNames[outcome]), 
        #         fontdict={'fontsize': 20, 'fontweight': 'bold'}, 
        #         ha="center"
        #     )

    # Hide unused axes
    #for j in range(len(plot_data), len(axes)):
    #    axes[j].axis('off')

    # Add a single legend for the whole figure
    #handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, title="bin", loc="upper right")

    # Save the figure
    fig.savefig("ConcentrationCurve.png", dpi=300)



#Plot Scatterplot with loess curve
def plotScatterplot(ax, combined_df, control, outcome, approach, palette, unique_bins):

    combined_df["error"] = abs(combined_df["pred"] - combined_df["true"])

    bin_colors = {bin: palette[i] for i, bin in enumerate(unique_bins)}

    # Compute percentiles for trimming
    x_min, x_max = combined_df["demographic_val"].quantile([0.05, 0.95])
    y_min, y_max = combined_df["error"].quantile([0.05, 0.95])
    
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
        hue="bin", 
        data=filtered_df,
        ax=ax,
        palette=bin_colors,
        alpha=.9,
        legend=False,
        s=18,
        edgecolor='none',
        hue_order=unique_bins
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



#Plot scatterplot with axes true vs model prediction for each datapoint
def plotScatterplotPredVsTrues(ax, combined_df, control, outcome, approach, palette, unique_bins):
    
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
    
    #draw scatter
    bin_colors = {bin: palette[i] for i, bin in enumerate(unique_bins)}
    sns.scatterplot(
        x="pred", 
        y="true", 
        hue="bin", 
        data=filtered_df,
        ax=ax,
        palette=bin_colors,
        alpha=.5,
        legend=False,
        s=18,
        edgecolor='none',
        hue_order=unique_bins
    )

    #Draw linear regressions for each split
    for bin, sub_df in filtered_df.groupby("bin"):

        slope, intercept, r_value, p_value, std_err = linregress(sub_df["pred"], sub_df["true"])
        
        transparent_color = mcolors.to_rgba(bin_colors[bin], alpha=0.4)
        sns.regplot(
            x="pred", 
            y="true", 
            data=sub_df, 
            ax=ax,
            scatter=False,   # Hide points
            color=transparent_color,  # Use bin-specific color
            line_kws={'lw': 5, 'alpha': 1},
            ci=None
        )

    #Draw the axes and title for the graph
    title = "True vs Predicted"
    wrapped_title = "\n".join(textwrap.wrap(title, width=CHART_TITLE_WIDTH))
    ax.set_title(wrapped_title)
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: round_to_2_sig_figs(y)))
    ax.set_xlabel('Predicted', labelpad=-1)
    ax.set_ylabel('True', labelpad=-3)



#Plot the concenctration curve/lorenz curve
def plotConcentrationCurve(ax, combined_df, control, outcome, approach, absolute=False):

    ypreds = combined_df["pred"]
    ytrues = combined_df["true"]
    demographics = combined_df["demographic_val"]
    absolute_diff = [abs(a - b) for a, b in zip(ypreds, ytrues)]

    rs = [v for _, v in sorted(zip(demographics, absolute_diff))]
    n = len(rs)

    cumulative_share_of_population = np.linspace(0, 1, n+1)
    cumulative_share_of_income = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    sumDeviationArea = npConcentrationCoefficientIntegrate(rs)

    # Plot the Lorenz curve
    lorenz_label = 'BCI: {:.1f}%'.format(sumDeviationArea * 100)
    ax.plot(cumulative_share_of_population, cumulative_share_of_income, label=lorenz_label, color='blue')
    ax.plot([0, 1], [0, 1], label='Line of Equality', color='red', linestyle='--')
    ax.fill_between(cumulative_share_of_population, cumulative_share_of_income, cumulative_share_of_population, color='blue', alpha=0.2)
    
    # title = "Concentration Curve of Error of {} vs {} (Summed Deviation Area: {:.3f} Absolute Deviation Area: {:.3f})".format(
    #     cleanNames[approach], cleanNames[control], sumDeviationArea, absoluteDeviationArea
    # )
    title = " BCI " + cleanNames[approach]
    wrapped_title = "\n".join(textwrap.wrap(title, width=CHART_TITLE_WIDTH))
    ax.set_title(wrapped_title)
    # Reduce the number of ticks to 3
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3))

    # Format tick labels to two decimal places
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: round_to_2_sig_figs(y)))
    ax.set_xlabel('Cumul. prop. of counties ')
    ax.set_ylabel('Cumul % of pred. error \n' + cleanNames[control] + " : " + cleanNames[outcome], labelpad=-5)
    ax.legend()



#Plot the lorenz curve with ks dotted line
def plotKSCurve(ax, combined_df, control, outcome, approach, absolute=False):

    ypreds = combined_df["pred"]
    ytrues = combined_df["true"]
    demographics = combined_df["demographic_val"]
    absolute_diff = [abs(a - b) for a, b in zip(ypreds, ytrues)]

    rs = [v for _, v in sorted(zip(demographics, absolute_diff))]
    n = len(rs)

    cumulative_share_of_population = np.linspace(0, 1, n+1)
    cumulative_share_of_income = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    ks_distances = np.abs(cumulative_share_of_income - cumulative_share_of_population)
    max_ks_index = np.argmax(ks_distances)
    max_ks_x = cumulative_share_of_population[max_ks_index]
    max_ks_y_lorenz = cumulative_share_of_income[max_ks_index]
    max_ks_y_equality = max_ks_x

    ax.plot(cumulative_share_of_population, cumulative_share_of_income, color='blue')
    ax.plot([0, 1], [0, 1], label='Line of Equality', color='red', linestyle='--')
    ax.fill_between(cumulative_share_of_population, cumulative_share_of_income, cumulative_share_of_population, color='blue', alpha=0.2)

    '''# Plot KS Test against Uniform(0,1)
    #sorted_values = np.sort(absolute_diff)  # Sort values
    ecdf = np.arange(1, len(rs) + 1) / len(rs)  # Empirical CDF
    uniform_cdf = uniform.cdf(rs, loc=np.average(rs), scale=0.01)  # Theoretical CDF (Uniform)

    ksStat, _ = kstest(rs, uniform.cdf, args=(0, 1))  # KS test

    # Plot empirical CDF
    ax.plot(cumulative_share_of_population, cumulative_share_of_income, label="Empirical CDF", color="green")
    ax.plot([0, 1], [0, 1], label='Line of Equality', color='red', linestyle='--')
    #ax.plot(cumulative_share_of_population, uniform_cdf, label="Uniform(0,1) CDF", color="orange", linestyle="--")'''

    # Highlight KS Statistic (max difference)
    ax.vlines(max_ks_x, max_ks_y_lorenz, max_ks_y_equality, color='black', linestyle='dashed', linewidth=2, label='KS Distance: {:.3f}'.format(KsTest(rs)))

    #max_diff_idx = np.argmax(np.abs(ecdf - uniform_cdf))
    #ax.vlines(rs[max_diff_idx], uniform_cdf[max_diff_idx], ecdf[max_diff_idx], colors='black', linestyle='dotted', label='KS Stat: {:.3f}'.format(ksStat))

    ax.set_title("KS Test Curve")
    ax.set_xlabel('Cumul. prop. of counties')
    ax.set_ylabel('Cumul pred. error', labelpad=-5)
    ax.legend()



# Perform KsTest
def KsTest(rs):
    n = len(rs)

    cumulative_share_of_population = np.linspace(0, 1, n+1)
    cumulative_share_of_income = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    ks_distances = np.abs(cumulative_share_of_income - cumulative_share_of_population)
    max_ks_index = np.argmax(ks_distances)

    return ks_distances[max_ks_index]



# Load all the prediction results from csv
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


# correct residual data from csv loading
def _correctResiduals(df):
    for index, row in df[df['test'].str.contains('Residual', na=False)].iterrows():
        #print(float(df.at[index, 'demographic']))
        if(df.at[index, 'demographic']):
            df.at[index, 'demographic_and_language'] = float(df.at[index, 'demographic_and_language']) + float(df.at[index, 'demographic'])
            df.at[index, 'true'] = float(df.at[index, 'true']) + float(df.at[index, 'demographic'])
    return df



# calculate correlations for results
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



def calcLikelihood(rs):
    
    # Compute mean and standard deviation of the data
    mu = np.mean(rs)
    sigma =np.std(rs)#mu / 3
    
    # Compute the likelihood
    log_likelihood = 0
    for x in rs:
        log_likelihood += np.log(1 / (sigma * np.sqrt(2 * np.pi))) - 0.5 * ((x - mu) / sigma) ** 2
        #print("TEST: ", np.log(1 / (sigma * np.sqrt(2 * np.pi))) - 0.5 * ((x - mu) / sigma) ** 2)
        #print("X: ", x)
        #print("TEST: ", (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2))
    
    return log_likelihood



def calcChiSquared(rs):
    expected = np.full_like(rs, 1/len(rs), dtype=float)#np.full_like(rs, np.mean(rs), dtype=float)
    rs = rs / np.sum(rs)
    #print("MEAN: ", rs)

    # Run the Chi-Squared test
    chi2, p, dof = chi2_contingency([rs, expected])[:3]

    # Output results
    print("Chi-Squared Statistic:", chi2)
    print("P-value:", p)
    print("Degrees of Freedom:", dof)

    # Interpret the result
    if p < 0.05:
        print("There is a significant difference between the counties.")
    else:
        print("There is no significant difference between the counties.")

    return chi2, p



def calcAndersonDarling(rs):
    n = len(rs)

    theoretical_cdf = np.linspace(0, 1, n+1)
    empirical_cdf = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)
    
    weighted_diff = (empirical_cdf - theoretical_cdf)**2 / (theoretical_cdf * (1 - theoretical_cdf))
    
    # Handle potential division by zero at the edges
    weighted_diff = np.nan_to_num(weighted_diff, nan=0.0, posinf=0.0, neginf=0.0)

    a_squared = n * np.sum(weighted_diff)
    
    return a_squared



def calcAndyDeviation(rs):
    rs = rs / np.sum(rs) #percent of total county that the bin is
    mu = 1/ len(rs) # expected
    product = 1
    for x in rs:
        product *= x * mu

    return product



def calcCrossEntropy(rs):
    rs = rs / np.sum(rs) #percent of total county that the bin is
    mu = 1/ len(rs) # expected
    product = 1
    for x in rs:
        product *= math.log(x) * mu

    return product



def calcJensenShannon(rs):
    # Create uniform distribution with same length as unique values in data
    rs = rs# / np.sum(rs)
    n = len(rs)
    uniform_dist = np.ones(n) * np.mean(rs) # / n
    
    # Convert to arrays for entropy calculation
    p = rs
    q = uniform_dist
    
    # Calculate the average distribution
    m = 0.5 * (p + q)
    
    # Calculate KL divergences
    kl_pm = entropy(p, m)
    kl_qm = entropy(q, m)
    
    # Jensen-Shannon divergence is average of KL divergences
    js_div = 0.5 * (kl_pm + kl_qm)
    
    return js_div



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
    return absoluteDeviationArea#, sumDeviationArea



def npConcentrationCoefficientIntegrate(rs):

    def f(x, n, e1, e2, u1, u2):
        return abs((n * (e2-e1-u2+u1))*x + e1 - u1)

    # Sort the data
    #rs = np.sort(rs)
    n = len(rs)

    # Calculate cumulative proportions
    cumulative_share_of_population = np.linspace(0, 1, n+1)
    cumulative_share_of_error = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    absoluteDeviationArea = 0
    for i in range(len(cumulative_share_of_population)-1):
        result, error = quad(lambda x: f(x, n, cumulative_share_of_error[i], cumulative_share_of_error[i+1], cumulative_share_of_population[i], cumulative_share_of_population[i+1]), 0, 1/n)
        absoluteDeviationArea += result
        #print("Test: ", n, cumulative_share_of_error[i]- cumulative_share_of_error[i+1], cumulative_share_of_population[i]- cumulative_share_of_population[i+1], result)

    # Compute the absolute deviation from the equality line
    #absoluteDeviationCurve = np.abs(cumulative_share_of_error - cumulative_share_of_population)
    #sumDeviationArea = 2 * ((np.trapz(cumulative_share_of_error, cumulative_share_of_population)) - .5)
    #absoluteDeviationArea = 2 * np.trapz(absoluteDeviationCurve, cumulative_share_of_population)
    return absoluteDeviationArea#, sumDeviationArea





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



#Bin functions to cut data into terciles
def create_bin_function(num_bins):
    def binning_function(df):
        bins = [df.min() - 1] + [df.quantile(i / num_bins) for i in range(1, num_bins)] + [df.max()]
        labels = list(range(0, num_bins))
        return pd.cut(df, bins=bins, labels=labels)
    return binning_function

binning_functions = {
    'logincomeHC01_VC85ACS3yr': create_bin_function(num_bins),
    'hsgrad': create_bin_function(num_bins),
    'forgnborn': create_bin_function(num_bins),
    'age': create_bin_function(num_bins),
    'is_female': lambda df: df.map({1: 2, 0: 1, None: 3}),
    'is_black': lambda df: df.map({1: 2, 0: 1, None: 3}),
    'individual_income': create_bin_function(num_bins),
}



def labelBins(df, control):
    for key in binning_functions:
        if key in control:
            masks = binning_functions[key](pd.to_numeric(df["demographic_val"], errors='coerce'))
            df['bin'] = masks
            return df
    raise ValueError("No matching mask function found for control: {}".format(control))



def bootstrapResampleBase(df, internal_metric = pearsonr, disp_metric=lambda x: 1-minMaxRatio(x), num_resamples = BOOTSTRAP_COUNT, bins =True, compareWithNull = False):
    ypreds = df["pred"]
    ytrues = df["true"]
    base_ypreds = df["base"]
    bin_ids = df["bin"]
    count_null_trials = 0

    if(bins):
        calculateMetric = lambda *args, **kwargs: calcMetricOnBins(*args, **kwargs)
    else:
        calculateMetric = lambda *args, **kwargs: calcMetricOnFullData(*args, **kwargs)

    disp_new = calculateMetric(ypreds, ytrues, disp_metric, bin_ids, internal_metric)
    for k in range(num_resamples):
        indices = np.random.choice(len(df), size=len(df), replace=True)

        bs_base_ypreds = base_ypreds.reset_index(drop=True).loc[indices].tolist() 
        bs_ytrues = ytrues.reset_index(drop=True).loc[indices].tolist() 
        bs_bin_ids = bin_ids.reset_index(drop=True).loc[indices].tolist() 
 
        disp_old = calculateMetric(bs_base_ypreds, bs_ytrues, disp_metric, bs_bin_ids, internal_metric)
        if abs(disp_base) <= abs(disp_new):
            count_null_trials += 1

    #print("AARON_DEBUG_RATIO: ", count_null_trials / num_resamples)
    return count_null_trials / num_resamples



def bootstrapResampleAlternative(df, internal_metric = pearsonr, disp_metric=lambda x: 1-minMaxRatio(x), num_resamples = BOOTSTRAP_COUNT, bins = True, compareWithNull = False):
    ypreds = df["pred"]
    ytrues = df["true"]
    base_ypreds = df["base"]
    bin_ids = df["bin"]
    count_null_trials = 0

    if(bins):
        calculateMetric = lambda *args, **kwargs: calcMetricOnBins(*args, **kwargs)
    else:
        calculateMetric = lambda *args, **kwargs: calcMetricOnFullData(*args, **kwargs)

    if compareWithNull:
        bin_ids = pd.Series(np.random.permutation(bin_ids.values), index=bin_ids.index)
        disp_base = calculateMetric(ypreds, ytrues, disp_metric, bin_ids, internal_metric)
    else:
        disp_base = calculateMetric(base_ypreds, ytrues, disp_metric, bin_ids, internal_metric)
    for k in range(num_resamples):
        indices = np.random.choice(len(df), size=len(df), replace=True)

        bs_ypreds = ypreds.reset_index(drop=True).loc[indices].tolist() 
        bs_ytrues = ytrues.reset_index(drop=True).loc[indices].tolist() 
        bs_bin_ids = bin_ids.reset_index(drop=True).loc[indices].tolist() 
 
        disp_new = calculateMetric(bs_ypreds, bs_ytrues, disp_metric, bs_bin_ids, internal_metric)
        if disp_base <= disp_new:
            count_null_trials += 1

    return count_null_trials / num_resamples



def bootstrapResampleBoth(df, internal_metric = pearsonr, disp_metric=lambda x: 1-minMaxRatio(x), num_resamples = BOOTSTRAP_COUNT, bins = True, compareWithNull = False):
    ypreds = df["pred"]
    ytrues = df["true"]
    base_ypreds = df["base"]
    bin_ids = df["bin"]
    demographics = df["demographic_val"]
    count_null_trials = 0

    if(bins):
        calculateMetric = lambda *args, **kwargs: calcMetricOnBins(*args, **kwargs)
    else:
        calculateMetric = lambda *args, **kwargs: calcMetricOnFullData(*args, **kwargs)


    for k in range(num_resamples):
        
        indices_new = np.random.choice(len(df), size=len(df), replace=True)
        bs_new_ypreds = ypreds.reset_index(drop=True).loc[indices_new].tolist() 
        bs_new_ytrues = ytrues.reset_index(drop=True).loc[indices_new].tolist() 
        bs_new_bin_ids = bin_ids.reset_index(drop=True).loc[indices_new].tolist() 
        bs_new_demographics = demographics.reset_index(drop=True).loc[indices_new].tolist() 

        disp_new = calculateMetric(bs_new_ypreds, bs_new_ytrues, demographics=bs_new_demographics, disp_metric=disp_metric, bin_ids=bs_new_bin_ids, internal_metric = internal_metric)
        if compareWithNull:
            if(bins):
                disp_original = calculateMetric(bs_new_ypreds, bs_new_ytrues, demographics=demographics, disp_metric=disp_metric, bin_ids=bs_new_bin_ids, internal_metric = internal_metric)
                #print("DISP Original: ", disp_original)
                bs_new_bin_ids = np.random.permutation(bs_new_bin_ids)
                disp_base = calculateMetric(bs_new_ypreds, bs_new_ytrues, demographics=demographics, disp_metric=disp_metric, bin_ids=bs_new_bin_ids, internal_metric = internal_metric)
                #print("DISP BASE: ", disp_base)
                
            else:
                indices_new = np.random.choice(len(df), size=len(df), replace=True)
                bs_new_ypreds = ypreds.reset_index(drop=True).loc[indices_new].tolist() 
                bs_new_ytrues = ytrues.reset_index(drop=True).loc[indices_new].tolist() 
                bs_new_bin_ids = bin_ids.reset_index(drop=True).loc[indices_new].tolist()
                disp_base = calculateMetric(bs_new_ypreds, bs_new_ytrues, demographics=demographics, disp_metric=disp_metric, bin_ids=bs_new_bin_ids, internal_metric = internal_metric)
        else:
            indices_base = np.random.choice(len(df), size=len(df), replace=True)
            bs_base_ypreds = base_ypreds.reset_index(drop=True).loc[indices_base].tolist() 
            bs_base_ytrues = ytrues.reset_index(drop=True).loc[indices_base].tolist() 
            bs_base_bin_ids = bin_ids.reset_index(drop=True).loc[indices_base].tolist() 
            bs_base_demographics = demographics.reset_index(drop=True).loc[indices_base].tolist() 
            disp_base = calculateMetric(bs_base_ypreds, bs_base_ytrues, demographics=bs_base_demographics, disp_metric=disp_metric, bin_ids=bs_base_bin_ids, internal_metric = internal_metric)
        #print("DISPARITIES: ", disp_new, disp_base)
        if abs(disp_base) <= abs(disp_new):
            count_null_trials += 1

    return count_null_trials / num_resamples



def calcMetricOnBins(ypreds, ytrues, demographics, disp_metric=lambda x: 1-minMaxRatio(x), bin_ids=None, internal_metric = pearsonr):
    numpyArrays = [np.array(var) for var in (ypreds, ytrues, demographics, bin_ids)]
    stacked_array = np.column_stack(numpyArrays)
    sorted_array = stacked_array[stacked_array[:, -2].argsort()]
    #print(sorted_array.shape)
    ypreds, ytrues, demographs, bin_ids = sorted_array.T

    #Split values by bin
    terc_ypreds = [[] for _ in range(num_bins)]
    terc_ytrues = [[] for _ in range(num_bins)]
    
    #sorted_array = stacked_array[stacked_array[:, -1].argsort()]
    temp = ""

    for i in range(len(ypreds)):
        this_bin = bin_ids[i].astype(int)
        #temp += str(sorted_array[i])
        terc_ypreds[this_bin].append(ypreds[i])
        terc_ytrues[this_bin].append(ytrues[i])

    
    last_column = sorted_array[:, -1]

    # Convert to a string of numbers
    last_column_str = "".join(map(str, last_column.astype(int)))
    #print("SORTED BINS: ", last_column_str)
    #print("Test: ", terc_ypreds[0][:5])

    #calculate the metric score per bin
    results = []
    for t in range(num_bins):
        cor = internal_metric(terc_ypreds[t], terc_ytrues[t])
        results.append(cor)

    return disp_metric(results)



def calcMetricOnFullData(ypreds, ytrues, demographics=None, disp_metric=lambda x: calculateGini(x), bin_ids=None, internal_metric = None):

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


