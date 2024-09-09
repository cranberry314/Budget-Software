#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# This file is part of Budget Software.
#
# Budget Software is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Budget Software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ExampleProject. If not, see
# <https://www.gnu.org/licenses/>.



@author: andrewfinn
This is version 1.0, I will make future improvements soon
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import json
import config.dictionaries as dicts

# Read settings from JSON configuration file
with open('config/settings.json') as f:
    settings = json.load(f)


# File paths
location = settings['location']
bank_statement_file = settings['bank_statement_file']
credit_card_file = settings['credit_card_file']

now = datetime.now()
cutoff_date = (now - pd.DateOffset(months=12)).replace(day=1)

# Function to process bank data
def process_bank_data(settings_file='config/settings.json'):
    """
    This function processes the bank data by reading a CSV file, classifying the transactions,
    calculating the monthly income, and returning the processed DataFrames for further analysis.

    Parameters:
    settings_file (str): Path to the JSON configuration file containing file locations.

    Returns:
    bank_df (DataFrame): The processed bank data.
    monthly_income_df (DataFrame): Monthly income calculated from the data.
    monthly_bank_debit_summary_df (DataFrame): Monthly debit summary excluding credit card transactions.
    """
    

    # Read and process bank data
    bank_df = pd.read_csv(f"{location}/{bank_statement_file}")
    bank_df['Transaction Date'] = pd.to_datetime(bank_df['Transaction Date'], format='%m/%d/%y')


    # Filter out transactions from the current month and year, and those older than 12 months
    bank_df = bank_df[
        (bank_df['Transaction Date'] < pd.Timestamp(now.replace(day=1))) & 
        (bank_df['Transaction Date'] >= cutoff_date)
    ]

    # Monthly income calculation
    income_df = bank_df[bank_df['Transaction Description'] == 'Deposit from CAPITAL ONE SERV REG.SALARY']
    monthly_income_df = income_df.groupby([income_df['Transaction Date'].dt.to_period('M')])['Transaction Amount'].sum().reset_index()
    monthly_income_df.rename(columns={'Transaction Date': 'YearMonth', 'Transaction Amount': 'Income'}, inplace=True)

    # Create end-of-month dates
    monthly_income_df['Date'] = monthly_income_df['YearMonth'].dt.to_timestamp() + pd.offsets.MonthEnd()
    monthly_income_df.sort_values(by='Date', ascending=True, inplace=True)

    # Classify bank transactions
    def classify_description(description):
        for key, category in dicts.DESCRIPTION_TO_CATEGORY.items():
            if key in description:
                return category
        return description

    bank_df['General Description'] = bank_df['Transaction Description'].apply(classify_description)

    # Add a Detailed Description
    def classify_detailed_description(description):
        """Classifies the description into a more detailed category."""
        for key, detail_category in dicts.DESCRIPTION_TO_DETAIL_CATEGORY.items():
            if key in description:
                return detail_category
        return description

    bank_df['Detailed Description'] = bank_df['Transaction Description'].apply(classify_detailed_description)

    # Remove Taxes and prepare data for plotting
    bank_df = bank_df[bank_df['General Description'] != 'Taxes']
    bank_df['Debit'] = bank_df.apply(lambda row: row['Transaction Amount'] if row['Transaction Type'] == 'Debit' else 0, axis=1)
    bank_df['Credit'] = bank_df.apply(lambda row: row['Transaction Amount'] if row['Transaction Type'] == 'Credit' else 0, axis=1)
    bank_df['YearMonth'] = bank_df['Transaction Date'].dt.to_period('M')

    # Filter out credit card transactions and prepare a monthly debit summary
    bank_debit_df = bank_df[bank_df['General Description'] != 'Credit Card'].drop(columns=['Credit'])
    monthly_bank_debit_summary_df = bank_debit_df.groupby(['YearMonth', 'General Description'])['Debit'].sum().reset_index()
    monthly_bank_debit_summary_df = monthly_bank_debit_summary_df[monthly_bank_debit_summary_df['Debit'] != 0]

    return bank_df, monthly_income_df, monthly_bank_debit_summary_df

# Call process_bank_data function
bank_df, monthly_income_df, monthly_bank_debit_summary_df = process_bank_data()


# Function to process credit card data
def process_credit_data(settings_file='config/settings.json'):
    """
    Processes credit card data by classifying descriptions into general and detailed categories,
    filtering transactions based on a 12-month cutoff, and summarizing debit transactions.

    Parameters:
    settings_file (str): Path to the settings JSON file containing the file locations.

    Returns:
    credit_df (DataFrame): Processed credit card transactions with categorized descriptions.
    monthly_credit_summary_df (DataFrame): Summary of debit transactions by 'YearMonth' and 'General Description'.
    credit_debit_df (DataFrame): Granular data is used in the identification of repeated transactions later
    """

    # Extract file paths from settings
    location = settings['location']
    credit_card_file = settings['credit_card_file']

    # Load credit card data
    credit_df = pd.read_csv(f"{location}/{credit_card_file}")

    # Classify General Description
    credit_description_to_category = dicts.CREDIT_DESCRIPTION_TO_CATEGORY
    def classify_credit_description(row):
        description = row['Description']
        for key, category in credit_description_to_category.items():
            if key in description:
                return category
        return row['Category']  # Default to original category if no match is found

    credit_df['General Description'] = credit_df.apply(classify_credit_description, axis=1)

    # Classify Detailed Description
    credit_description_to_detail_category = dicts.CREDIT_DESCRIPTION_TO_DETAIL_CATEGORY
    def classify_detailed_description(row):
        description = row['Description']
        for key, detail_category in credit_description_to_detail_category.items():
            if key in description:
                return detail_category
        return row['Category']  # Default to original category if no match is found

    credit_df['Detailed Description'] = credit_df.apply(classify_detailed_description, axis=1)

    # Get the current date and calculate the cutoff date (12 months ago)
    now = datetime.now()
    cutoff_date = (now - pd.DateOffset(months=12)).replace(day=1)

    # Convert 'Transaction Date' to datetime format
    credit_df['Transaction Date'] = pd.to_datetime(credit_df['Transaction Date'])

    # Filter transactions that are older than 12 months and exclude transactions from the current month
    credit_df = credit_df[
        (credit_df['Transaction Date'] < pd.Timestamp(now.replace(day=1))) & 
        (credit_df['Transaction Date'] >= cutoff_date)
    ]

    # Add 'YearMonth' column for easy grouping
    credit_df['YearMonth'] = credit_df['Transaction Date'].dt.to_period('M')

    # Drop 'Credit' column to focus on debit transactions
    credit_debit_df = credit_df.drop(columns=['Credit'])

    # Summarize debit transactions by 'YearMonth' and 'General Description'
    monthly_credit_summary_df = credit_debit_df.groupby(['YearMonth', 'General Description'])['Debit'].sum().reset_index()

    # Remove rows where Debit equals 0
    monthly_credit_summary_df = monthly_credit_summary_df[monthly_credit_summary_df['Debit'] != 0]

    return credit_df, monthly_credit_summary_df, credit_debit_df

# Call process_credit_data
# we needs to return the very granular credit_debit_df because it is used in the identification of repeated transactions
credit_df, monthly_credit_summary_df, credit_debit_df = process_credit_data()



# Combine summaries and add expense types
monthly_summary_df = pd.concat([monthly_bank_debit_summary_df, monthly_credit_summary_df], ignore_index=True)


fixed_vs_variable_dict = {
    'Mortgage': 'Fixed', 'Childcare': 'Fixed', 'Exercise': 'Fixed', 'Internet': 'Fixed', 'Cable': 'Fixed',
    'Insurance': 'Fixed', 'Phone': 'Fixed', 'Phone/Cable': 'Fixed', 'Dining': 'Variable', 'Entertainment': 'Variable',
    'Groceries': 'Variable', 'Health Care': 'Variable', 'Merchandise': 'Variable', 'Other': 'Variable',
    'Other Services': 'Variable', 'Auto/Transport': 'Variable', 'Other Travel': 'Variable',
    'ATM': 'Variable', 'Lodging': 'Variable', 'Home Repairs': 'Variable', 'Fee/Interest Charge': 'Variable', 'Taxes': 'Variable'
}
monthly_summary_df['Fixed vs Variable'] = monthly_summary_df['General Description'].map(fixed_vs_variable_dict)

# Summarize expenses
summary_df = monthly_summary_df.groupby(['YearMonth', 'Fixed vs Variable'])['Debit'].sum().unstack(fill_value=0)

# Get unique months
months = monthly_summary_df['YearMonth'].unique()

# Plot data
fig, ax = plt.subplots(figsize=(16, 12))
summary_df.plot(kind='bar', stacked=True, ax=ax, color=['#000000', '#1f77b4'], width=0.8)

# Add labels and title
ax.set_xlabel('Month')
ax.set_ylabel('Amount')
ax.set_title('Monthly Fixed and Variable Expenses')
plt.xticks(rotation=45)
plt.subplots_adjust(left=0.1, right=1.25, top=0.85, bottom=0.55)

# Overlay donut charts on top of each bar

# Define a fixed color palette with 20 colors
fixed_colors = plt.get_cmap('tab20').colors
# Map description to color
description_colors = {
    desc: fixed_colors[i % len(fixed_colors)]
    for i, desc in enumerate(monthly_summary_df['General Description'].unique())
}

for i, month in enumerate(months):
    # Filter data for the current month
    data_for_month = monthly_summary_df[monthly_summary_df['YearMonth'] == month]

    # Calculate sizes and labels
    sizes = data_for_month.groupby('General Description')['Debit'].sum()
    total = sizes.sum()
    
    # Sort data by size (largest to smallest)
    sorted_sizes = sizes.sort_values(ascending=False)
    sorted_labels = sorted_sizes.index
    sorted_colors = [description_colors[label] for label in sorted_labels]
    
    # Plot the donut chart below the bar
    donut_ax = fig.add_axes([0.12 + i * 0.094, 0.40, 0.07, 0.1], aspect='equal')
    wedges, _ = donut_ax.pie(
        sorted_sizes,
        labels=None,
        startangle=140,
        wedgeprops=dict(width=0.3, edgecolor='w'),
        colors=sorted_colors
    )

    # Create legend for donut chart
    donut_legend_labels = [f'{label}: {size:.2f} ({(size / total) * 100:.1f}%)' for label, size in zip(sorted_labels, sorted_sizes)]
    donut_ax.legend(wedges, donut_legend_labels, title="Categories", loc="upper right", bbox_to_anchor=(1.3, -0.15), fontsize='7')

# Plot income data
ax2 = ax.twinx()
ax2.plot(monthly_income_df['YearMonth'].astype(str), monthly_income_df['Income'], marker='D', color='g', linestyle='-', linewidth=2, markersize=8, label='Monthly Income')
ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize='small')

# Hide X-axis on ax2
ax2.spines['bottom'].set_visible(False)  # Hide the X-axis line
ax2.xaxis.set_visible(False)  # Hide the X-axis ticks and labels

# Save plot to PDF
#plt.savefig(f"{location}/monthly_expenses_and_income.pdf", format='pdf', bbox_inches='tight')
#plt.show()


#
#
# Page 2 of the graph, this is to highlight repeated expenses, unnecessary expenses, and necessary expenses
# Aggregate the data by YearMonth and Detailed Description to calculate sum and mean
#
#

# Detailed summary lives up here till I can get things sorted
# Group by 'YearMonth' and 'Detailed Description' and sum the 'Debit' values
bank_grouped = bank_df.groupby(['YearMonth', 'Detailed Description'])['Debit'].sum().reset_index()
credit_grouped = credit_df.groupby(['YearMonth', 'Detailed Description'])['Debit'].sum().reset_index()

# Merge the bank and credit data on 'YearMonth' and 'Detailed Description'
merged_df = pd.merge(bank_grouped, credit_grouped, on=['YearMonth', 'Detailed Description'], how='outer', suffixes=('_bank', '_credit'))

# Fill NaN values with 0 for any missing data
merged_df.fillna(0, inplace=True)

# Sum the 'Debit' amounts from both bank and credit
merged_df['Total'] = merged_df['Debit_bank'] + merged_df['Debit_credit']

# Select relevant columns
monthly_detailed_summary_df = merged_df[['YearMonth', 'Detailed Description', 'Total']]



# Start a new figure for the second page
fig2, axs = plt.subplots(2, 2, figsize=(16, 12))

# Repeated Charges, top left
# Total number of unique months
total_months = credit_debit_df['YearMonth'].nunique()

# Find the most recent month in the data
most_recent_month = credit_debit_df['YearMonth'].max()

# Group by Description and Debit, count unique months
grouped_df = credit_debit_df.groupby(['Description', 'Debit']).agg({'YearMonth': 'nunique'}).reset_index()
grouped_df.rename(columns={'YearMonth': 'Unique Months'}, inplace=True)

# Filter charges that repeat for at least 6 months and are less than $25.00
repeating_charges_df = grouped_df[(grouped_df['Unique Months'] >= 6) & (grouped_df['Debit'] < 25)]

# Identify charges that also have a value in the most recent month
recent_month_charges = credit_debit_df[
    (credit_debit_df['YearMonth'] == most_recent_month) & 
    (credit_debit_df['Description'].isin(repeating_charges_df['Description']))
]

# Filter repeating charges to include only those present in the most recent month
filtered_repeating_charges_df = repeating_charges_df[
    repeating_charges_df['Description'].isin(recent_month_charges['Description'])
]

# Sort charges by Debit from largest to smallest
sorted_repeating_charges_df = filtered_repeating_charges_df.sort_values(by='Debit', ascending=False)
total_sum_repeating_charges = sorted_repeating_charges_df['Debit'].sum()

# Display the filtered DataFrame
print("Repeating Charges for at least 6 months, less than $25.00, and present in the most recent month:")
print(sorted_repeating_charges_df)
print(total_sum_repeating_charges)

# Print the repeating charges table on the top left corner of the second page
axs[0, 0].axis('off')
table = axs[0, 0].table(cellText=sorted_repeating_charges_df.values,
                        colLabels=sorted_repeating_charges_df.columns,
                        cellLoc='center', loc='center', fontsize=10)
table.auto_set_column_width(col=list(range(len(sorted_repeating_charges_df.columns))))

# Create a nested grid within axs[0, 1] for three charts
# Create a nested grid for three plots in the top-right corner
from matplotlib.gridspec import GridSpec

gs = GridSpec(2, 2, figure=fig2)  # Create a new grid for the existing figure
inner_grid = gs[0, 1].subgridspec(3, 1, hspace=1.5)  # Divide the top-right corner into a 3x1 grid


import seaborn as sns

# Create Uber dataframe for graphing
uber_monthly_totals = monthly_detailed_summary_df[monthly_detailed_summary_df['Detailed Description'] == 'Uber'].reset_index()
uber_monthly_totals['Date'] = uber_monthly_totals['YearMonth'].dt.to_timestamp()

# Plot "Uber" data with Seaborn (scatter points and regression line)
ax_21 = fig2.add_subplot(inner_grid[0])
sns.regplot(x=uber_monthly_totals['Date'].astype(int), y=uber_monthly_totals['Total'], 
            scatter_kws={'s': 50, 'color': 'r'}, line_kws={'color': 'r'}, ax=ax_21)
ax_21.set_title('Uber Monthly Totals')
ax_21.set_xlabel('Month')
ax_21.set_ylabel('Total Amount ($)')
ax_21.grid(True)

# Create Groceries dataframe for graphing
groceries_monthly_totals = monthly_detailed_summary_df[monthly_detailed_summary_df['Detailed Description'] == 'Groceries'].reset_index()
groceries_monthly_totals['Date'] = groceries_monthly_totals['YearMonth'].dt.to_timestamp()

# Plot "Groceries" data with Seaborn (scatter points and regression line)
ax_22 = fig2.add_subplot(inner_grid[1])
sns.regplot(x=groceries_monthly_totals['Date'].astype(int), y=groceries_monthly_totals['Total'], 
            scatter_kws={'s': 50, 'color': 'b'}, line_kws={'color': 'b'}, ax=ax_22)
ax_22.set_title('Groceries Monthly Totals')
ax_22.set_xlabel('Month')
ax_22.set_ylabel('Total Amount ($)')
ax_22.grid(True)

# Create Dining dataframe for graphing
dining_monthly_totals = monthly_detailed_summary_df[monthly_detailed_summary_df['Detailed Description'] == 'Dining'].reset_index()
dining_monthly_totals['Date'] = dining_monthly_totals['YearMonth'].dt.to_timestamp()

# Plot "Dining" data with Seaborn (scatter points and regression line)
ax_33 = fig2.add_subplot(inner_grid[2])
sns.regplot(x=dining_monthly_totals['Date'].astype(int), y=dining_monthly_totals['Total'], 
            scatter_kws={'s': 50, 'color': 'g'}, line_kws={'color': 'g'}, ax=ax_33)
ax_33.set_title('Dining Monthly Totals')
ax_33.set_xlabel('Month')
ax_33.set_ylabel('Total Amount ($)')
ax_33.grid(True)



#import seaborn as sns

# Load the Anscombe's quartet dataset
anscombe_df = sns.load_dataset('anscombe')

# Plot the second example (dataset II) in the bottom-left subplot with regression line
data3 = anscombe_df[anscombe_df['dataset'] == 'II']
sns.regplot(x='x', y='y', data=data3, ax=axs[1, 0], ci=None, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
axs[1, 0].set_title('Anscombe Example 2')
axs[1, 0].legend(['Data points', 'Regression line'])

# Plot the third example (dataset III) in the bottom-right subplot with regression line
data4 = anscombe_df[anscombe_df['dataset'] == 'III']
sns.regplot(x='x', y='y', data=data4, ax=axs[1, 1], ci=None, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
axs[1, 1].set_title('Anscombe Example 3')
axs[1, 1].legend(['Data points', 'Regression line'])



# Save plot to PDF
# fig2.savefig(f"{location}/monthly_expenses_and_income.pdf", format='pdf')
plt.show()


