import numpy as np

def mean_income(df):
    return df['income'].mean()

def median_income(df):
    return df['income'].median()

def p90_income(df):
    return np.percentile(df['income'], 90)

def weighted_average_income_by_city_country(df):
    # Compute total income per city
    city_totals = df.groupby(['country', 'city'])['income'].sum().rename('city_income_total')
    
    # Compute total income per country
    country_totals = df.groupby('country')['income'].sum().rename('country_income_total')
    
    # Merge totals back into main dataframe
    df = df.merge(city_totals, on=['country', 'city'])
    df = df.merge(country_totals, on='country')
    
    # Compute weight: city's total income / country's total income
    df['weight'] = df['city_income_total'] / df['country_income_total']
    
    # Now calculate weighted average using these weights
    weighted_avg = np.average(df['income'], weights=df['weight'])
    
    return weighted_avg
