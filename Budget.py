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
credit_card_file = settings['credit_card_file']
bank_statement_file = settings['bank_statement_file']

# Read and process bank data
bank_df = pd.read_csv(f"{location}/{bank_statement_file}")
bank_df['Transaction Date'] = pd.to_datetime(bank_df['Transaction Date'], format='%m/%d/%y')

# Filter out current month and year transactions
now = datetime.now()
current_month, current_year = now.month, now.year
bank_df = bank_df[~((bank_df['Transaction Date'].dt.month == current_month) & 
                     (bank_df['Transaction Date'].dt.year == current_year))]

# Monthly income calculation
income_df = bank_df[bank_df['Transaction Description'] == 'Deposit from CAPITAL ONE SERV REG.SALARY']
monthly_income_df = income_df.groupby([income_df['Transaction Date'].dt.to_period('M')])['Transaction Amount'].sum().reset_index()
monthly_income_df.rename(columns={'Transaction Date': 'YearMonth', 'Transaction Amount': 'Income'}, inplace=True)

# Create end-of-month dates
monthly_income_df['Date'] = monthly_income_df['YearMonth'].dt.to_timestamp() + pd.offsets.MonthEnd()
monthly_income_df.sort_values(by='Date', ascending=True, inplace=True)

# Classify bank transactions
description_to_category = dicts.DESCRIPTION_TO_CATEGORY

def classify_description(description):
    for key, category in description_to_category.items():
        if key in description:
            return category
    return description

bank_df['General Description'] = bank_df['Transaction Description'].apply(classify_description)

# Remove Taxes and prepare data for plotting
bank_df = bank_df[bank_df['General Description'] != 'Taxes']
bank_df['Debit'] = bank_df.apply(lambda row: row['Transaction Amount'] if row['Transaction Type'] == 'Debit' else 0, axis=1)
bank_df['Credit'] = bank_df.apply(lambda row: row['Transaction Amount'] if row['Transaction Type'] == 'Credit' else 0, axis=1)
bank_df['YearMonth'] = bank_df['Transaction Date'].dt.to_period('M')

bank_debit_df = bank_df[bank_df['General Description'] != 'Credit Card'].drop(columns=['Credit'])
monthly_bank_debit_summary_df = bank_debit_df.groupby(['YearMonth', 'General Description'])['Debit'].sum().reset_index()
monthly_bank_debit_summary_df = monthly_bank_debit_summary_df[monthly_bank_debit_summary_df['Debit'] != 0]

# Process credit card data
credit_df = pd.read_csv(f"{location}/{credit_card_file}")
credit_description_to_category = dicts.CREDIT_DESCRIPTION_TO_CATEGORY

def classify_credit_description(row):
    description = row['Description']
    for key, category in credit_description_to_category.items():
        if key in description:
            return category
    return row['Category']

credit_df['General Description'] = credit_df.apply(classify_credit_description, axis=1)
credit_df['Transaction Date'] = pd.to_datetime(credit_df['Transaction Date'])
credit_df = credit_df[~((credit_df['Transaction Date'].dt.month == current_month) & 
                         (credit_df['Transaction Date'].dt.year == current_year))]
credit_df['YearMonth'] = credit_df['Transaction Date'].dt.to_period('M')

# Combine and summarize debit data
credit_debit_df = credit_df.drop(columns=['Credit'])
monthly_credit_summary_df = credit_debit_df.groupby(['YearMonth', 'General Description'])['Debit'].sum().reset_index()
monthly_credit_summary_df = monthly_credit_summary_df[monthly_credit_summary_df['Debit'] != 0]

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
plt.savefig(f"{location}/monthly_expenses_and_income.pdf", format='pdf', bbox_inches='tight')
plt.show()
