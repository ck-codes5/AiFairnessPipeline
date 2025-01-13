import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for headless environments
import matplotlib.pyplot as plt
import re
import numpy as np
#import seaborn as sns



FILTER_OUT_CONTROLS_ONLY=False
FILTER_OUT_LANGUAGE_ONLY=False
INCLUDE_VALUES = True
AVERAGE_METHODS = False
COL = 'r'


def main():
    # File path to your CSV
    #file_path = 'original_printDS4UD_5_Outcomes.csv'#'original_printCTLB_1grams_predictionsFixed.csv'#'original_printDS4UD_5_Outcomes.csv'
    #file_path = 'original_printCTLB_1grams_SingleControls.csv'
    file_path = 'original_printDS4UD_5_Outcomes_SingleControlAgeFemale.csv'

    # Parse the CSV file into sections
    sections = parse_csv(file_path)

    # Plot all sections
    plot_all_sections(sections)

    

def parse_csv(file_path):
    """
    Parse the CSV file with repeated sections. Each section has:
    - A title row
    - Column header row
    - A series of data rows
    """
    sections = []
    
    # Use pandas to read the entire file without manually splitting lines
    # Read the CSV file with no specific constraints on columns
    
    # Assume 47 is the maximum number of columns found in the file
    column_names = ['col ' + str(i) for i in range(48)]
    
    # Read the CSV with predefined column names
    data = pd.read_csv(file_path, header=None, skip_blank_lines=True, names=column_names, engine='python')

    i = 0
    while i < len(data):
        title = data.iloc[i][0].strip()  # First row of section is the title
        outcomes = data.iloc[i][1].strip()  # First row of section is the title
        column_headers = list(data.iloc[i+1])  # Second row is the column names
        data_rows = []
        i += 2
        
        while i < len(data) and not data.iloc[i][0].startswith('row') and data.iloc[i][0].isdigit():  # Assuming empty row separates sections
            data_rows.append(list(data.iloc[i]))
            i += 1

        # Convert the data into a DataFrame
        df = pd.DataFrame(data_rows, columns=column_headers)
        print("headers: ", column_headers)
        df['Title'] = title  # Add the title as a column for easy reference
        df['allOutcomes'] = outcomes
        sections.append(df)
    
    return sections


'''
def color_code(val, min_val, max_val):
    """
    Normalize the value and return a color from red to green.
    """
    norm_val = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    scaler = .8
    red = int(255 * ((1 - norm_val) * scaler + (1-scaler)))
    print("RED: ", red)
    green = int(255 * (norm_val * scaler + (1-scaler)))
    return (red/255, green/255, .2)

def plot_all_sections(sections):
    """
    For each section in the CSV file, plot the 'r' column from all sections in a table with color-coded cells.
    """
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to combine all data
    
    for idx, df in enumerate(sections):
        if FILTER_OUT_CONTROLS_ONLY:
            df = df[df['model_controls'].str[-1] == '1']

        if FILTER_OUT_LANGUAGE_ONLY:
            df = df[~(df['model_controls'].str[-3] == '(')]

        title = df['Title'].iloc[0]
        outcomes = df['allOutcomes'].iloc[0].translate({ord(c): None for c in "[]'"})

        if COL in df.columns:
            df_r = df[[COL]].copy()  # Select only the 'r' column
            df_r[COL] = pd.to_numeric(df_r[COL], errors='coerce')
            df_r = df_r.dropna()

            if not df_r.empty:
                df_r['TestType'] = ''.join(word[:3] for word in title.split()[:-1] if word)
                df_r['Title'] = '_'.join(max((sub.strip() for sub in word.split("_") if sub), key=len, default='')[:3] for word in outcomes.split(",") if word)
                df_r['outcome'] = df['outcome']
                df_r['controls'] = df['model_controls']
                df_r['N'] = df['test_size'].astype(int) + df['train_size'].astype(int)
                df_r['num_features'] = df['num_features']
                df_r['label'] = df_r['Title'] + "_" + df_r['controls'].astype(str).apply(lambda x: '_'.join(word[:2] for word in re.sub(r'["\'()_]', '', x).split(",") if word))
                combined_df = pd.concat([combined_df, df_r], ignore_index=True)

            else:
                print("No valid data in column for section: {}".format(title))

        else:
            print("Column not found in section: {}".format(title))
    
    combined_df.to_csv('clean_' + file_path, index=False)

    if AVERAGE_METHODS:
        combined_df = combined_df.groupby('TestType', as_index=False).mean()
        combined_df['Title'] = ""
    
    if not combined_df.empty:
        print("COMB: ", combined_df)
        try:
            
            combined_df['identifier'] = combined_df['TestType'] + '_' + combined_df['label']

            # Step 2: Create a pivot table with TestType and controls as columns
            pivot_df = combined_df.pivot(index='outcome', columns='identifier', values='r')


            #pivot_df = combined_df.pivot_table(index='outcome', columns=['TestType', 'controls'], values=COL, aggfunc='first')

            
        except Exception as e:
            print("Error during pivoting:", e)
            return
        

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=pivot_df.values,
                         rowLabels=pivot_df.index,
                         colLabels=pivot_df.columns,
                         cellLoc='center',
                         loc='center')
        print("COMB: ", combined_df.columns)
        print("TABLE: ", combined_df)

        table.auto_set_font_size(False)  # Disable automatic font size adjustment
        table.set_fontsize(10)  # Set the desired font size

        # Optionally, you can also set the size of the columns
        table.scale(1.2, 1.2)  # Adjust the scale if needed

        # Color-code cells
        for i in range(len(pivot_df.index)):
            row_values = pivot_df.iloc[i, :]
            min_val = row_values.min()
            max_val = row_values.max()
            for j in range(len(pivot_df.columns)):
                value = pivot_df.iloc[i, j]
                if not pd.isnull(value):
                    color = color_code(value, min_val, max_val)
                    table.get_celld()[(i+1, j)].set_facecolor(color)
                    table.get_celld()[(i+1, j)].set_text_props(color='white' if value < (min_val + max_val) / 2 else 'black')
                else:
                    table.get_celld()[(i+1, j)].set_facecolor('white')

        for (i, j), cell in table.get_celld().items():
            if i > 0 and j >= 0:  # Avoid row and column headers
                try:
                    cell_text = cell.get_text().get_text()
                    cell.get_text().set_text("{:.3f}".format(float(cell_text)))  # Format the text to 3 decimals
                except ValueError:
                    continue  # Skip if the text isn't a number
        
        
        plt.savefig('colored_table.png')
        plt.close()
        print("Color-coded table saved as 'colored_table.png'")

    else:
        print("No valid data found to display.")
'''



def plot_all_sections(sections):
    """
    For each section in the CSV file, plot the 'r' column from all sections on a single graph if it exists.
    """
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to combine all data
    
    for idx, df in enumerate(sections):

        if FILTER_OUT_CONTROLS_ONLY:
            df = df[df['model_controls'].str[-1] == '1']

        if FILTER_OUT_LANGUAGE_ONLY:
            df = df[~((df['model_controls'].str[-3] == '(') & (df['Title'] != "Regression Test"))]
            
            df.loc[~((df['model_controls'].str[-3] == '(')) & (df['Title'] == "Regression Test"), 'Title'] = 'Regression Add Controls Test'
            


        title = df['Title'].iloc[0]  # Get the section title

        outcomes = df['allOutcomes'].iloc[0]
        chars_to_remove = "[]'"
        outcomes = ''.join(c for c in outcomes if c not in chars_to_remove)

        # Check if the column 'r' exists in the DataFrame
        if COL in df.columns:

            df_r = df[[COL]]  # Select only the 'r' column

            # Convert the 'r' column to numeric, ignore errors
            df_r[COL] = pd.to_numeric(df_r[COL], errors='coerce')
            
            # Drop rows where 'r' is NaN
            df_r = df_r.dropna()
            
            if not df_r.empty:
                # Add a column for the title to identify each section
                print("COLUMNS: ", df.columns)
                df_r['TestType'] = df['Title'].str.replace(' Test', '', case=False, regex=False)
                df_r['Title'] = '_'.join(max((sub.strip() for sub in word.split("_") if sub), key=len, default='')[:3] for word in outcomes.split(",") if word)
                df_r['MSE'] = df['mse']
                df_r['MAE'] = df['mae']
                df_r['outcome'] = df['outcome']
                df_r['controls'] = df['model_controls']
                df_r['N'] = df['test_size'].astype(int) + df['train_size'].astype(int)
                df_r['num_features'] = df['num_features']
                df_r['label'] = df_r['Title'] + "_" + df_r['controls'].astype(str).apply(lambda x: '_'.join(word[:5] for word in re.sub(r'["\'()_]', '', x).split(",") if word))
                combined_df = pd.concat([combined_df, df_r], ignore_index=True)
                
            else:
                print("No valid data in column for section: {}".format(title))

        else:
            print("Column not found in section: {}".format(title))



    combined_df.to_csv('clean_' + file_path, index=False)

    csv_df = combined_df

    csv_df = csv_df.pivot_table(
            index=['controls', 'outcome'],
            columns=['TestType'],
            values=['r', 'MSE', 'N', 'MAE', 'num_features', 'label', 'controls'],
            aggfunc='first'
        )
    csv_df.to_csv('clean_' + file_path)

    if AVERAGE_METHODS:
        combined_df = combined_df.groupby('TestType', as_index=False).mean()
        combined_df['Title'] = ""


    if not combined_df.empty:

        manual_colors = {
            'Regression': '#b4bfd1',         # Example color for 'Reg'
            'Regression Add Controls': '#687282',   # Example color for 'ResConReg'
            'Residualized Controls Regression': '#F5B841',   # Example color for 'ResConReg'
            'Factor Adaptation Regression': '#931621',   # Example color for 'FacAdaReg'
            'Residualized Factor Adaptation Regression': '#2E294E' # Example color for 'ResFacAdaReg'
        }
        shortenedNames = {
            'Regression': 'Lang only',         # Example color for 'Reg'
            'Regression Add Controls': 'Lang + Cont',   # Example color for 'ResConReg'
            'Residualized Controls Regression': 'Res Cont',   # Example color for 'ResConReg'
            'Factor Adaptation Regression': 'Fac Adapt',   # Example color for 'FacAdaReg'
            'Residualized Factor Adaptation Regression': 'Res Fac Adapt' # Example color for 'ResFacAdaReg'
        }
        shortenedOutcomes = {
            'avg_audit10_score': 'AD',         # Example color for 'Reg'
            'avg_neg_affect_score': 'NA',   # Example color for 'ResConReg'
            'avg_phq9_score': 'PHQ9',   # Example color for 'ResConReg'
            'avg_pos_affect_score': 'PA',   # Example color for 'FacAdaReg'
            'avg_pss_score': 'PSS', # Example color for 'ResFacAdaReg'
            'heart_disease' : 'heart disease',         # Example color for 'Reg'
            'life_satisfaction': 'life satisfaction',   # Example color for 'ResConReg'
            'perc_fair_poor_health': 'fair poor health',   # Example color for 'ResConReg'
            'suicide': 'suicide',   # Example color for 'FacAdaReg'
        }

        custom_order = ['Regression', 'Regression Add Controls', 'Residualized Controls Regression', 'Factor Adaptation Regression', 'Residualized Factor Adaptation Regression']
        combined_df['TestType'] = pd.Categorical(combined_df['TestType'], categories=custom_order, ordered=True)
        try:
            combined_df = combined_df.sort_values(by=['outcome', 'TestType'])
        except:
             combined_df = combined_df.sort_values(by=['TestType'], ascending=True)
        num_columns = len(combined_df['TestType'].unique())
        figure_width = max(num_columns * 2, 3)  # Adjust the scaling factor as needed

        colors_for_plot = combined_df['TestType'].map(manual_colors)
        unique_titles = combined_df['TestType'].unique()
        color_map = plt.get_cmap('tab20')  # You can choose a different colormap
        colors = {title: color_map(i / len(unique_titles)) for i, title in enumerate(unique_titles)}
       
        # Set bar width
        bar_width = .8  # Adjust this value to reduce/increase the width of individual bars
        # Increase spacing between groups of bars
        spacing = -.9  # Adjust this to increase space between groups of bars

        # Create an array of x positions for each bar
        num_groups = len(combined_df['TestType'].unique())
        x = np.arange(len(combined_df['outcome'].unique()))  # Create x positions based on unique outcomes
        group_width = bar_width + spacing  # Total width for each group of bars

        ax = combined_df.plot(kind='bar', x='outcome', y=COL, stacked=False, color=colors_for_plot, width=bar_width)#[colors[title] for title in combined_df['TestType']])
        
        # Create x positions for the bars
        x = 2.3 * bar_width  + np.arange(len(combined_df['outcome'].unique())) * (bar_width * 6.25)  # Position for each outcome

        # Plotting the bars
        ax = combined_df.plot(kind='bar', x='outcome', y=COL, stacked=False, color=colors_for_plot, width=bar_width)

        # Set x-ticks to reflect the adjusted positions
        ax.set_xticks(x)  # Set x-ticks to match the new x positions
        short_outs = [shortenedOutcomes.get(title, title) for title in combined_df['outcome'].unique()]
        ax.set_xticklabels(short_outs, rotation=180, ha='center', fontsize=12)  # Center the labels

        # Adjust the positions of the bars
        for i, bar in enumerate(ax.patches):
            bar.set_x(bar.get_x() + (i % num_groups) * group_width)  # Shift bars within groups

        # Create a custom legend with labels for each color
        handles = [plt.Line2D([0], [0], color=manual_colors[title], lw=18) for title in unique_titles]
        shortened_labels = [shortenedNames.get(title, title) for title in unique_titles]
        ax.legend(handles, shortened_labels, title='', handlelength=.5, handleheight=1, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=22)

        if INCLUDE_VALUES:
            for bar in ax.patches:
                height = bar.get_height()
                formatted_height = '{:.2f}'.format(height).lstrip('0')
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, formatted_height, ha='center', va='bottom', fontsize=14)
        

        # Set tick marks and labels
        #tick_positions = np.arange(0, len(combined_df['outcome'].unique()), 5)  # Get positions for every 5 bars
        #tick_labels = combined_df['outcome'].unique()[tick_positions]  # Get labels for those positions

        # Apply the tick positions and labels
        #ax.set_xticks(tick_positions)
        #ax.set_xticklabels(tick_labels, rotation=0)  # Rotate labels to horizontal


        plt.gcf().set_size_inches(figure_width, 8)
        #plt.ylim(.35, .9)
        plt.ylim(.1, .8)
        plt.title('')
        ax.set_ylabel('Pearson r', fontsize=24)
        ax.set_xlabel('', fontsize=24)
        plt.xticks(rotation=90)
        ax.tick_params(axis='y', labelsize=22)  # Increase y-tick label font size
        ax.tick_params(axis='x', rotation=0, labelsize=24)
        plt.tight_layout(rect=[0, 0, 1, 1])
        
        
        # Save the combined plot to a file
        plt.savefig('combined_plot2.png')

        plt.close()  # Close the plot to free memory

        # Save the combined DataFrame to a CSV file
        
    else:
        print("No valid data found to plot.")



def unique_values_ordered(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result



if __name__ == "__main__":
    main()