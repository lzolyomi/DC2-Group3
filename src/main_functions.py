import pandas as pd
import numpy as np 
import datetime

def preproccessing_transactions(df_transactions):
    df_transactions['day'] = pd.to_datetime(df_transactions['day'])
    df_transactions['time'] = pd.to_datetime(df_transactions['time'])
    df_transactions["hour"] = pd.to_datetime(df_transactions['time'], format='%H:%M:%S').dt.hour
    df_transactions["nday"] = pd.to_datetime(df_transactions["day"], dayfirst=True).dt.dayofyear
    df_transactions['week'] = df_transactions['day'].dt.strftime('%U')
    df_transactions['week'] = pd.to_numeric(df_transactions['week'], errors = 'coerce')
    return df_transactions

def date_converter(row):
    converted_date = datetime.datetime(2018, 1, 1) + datetime.timedelta(row["day"])
    return converted_date

def week_of_year(row):
    week_nr = datetime.date(row["date"].year, row["date"].month, row["date"].day).isocalendar()[1]
    return week_nr

def preproccessing_inventory(df_transactions):
    df_transactions["date"] = df_transactions.apply(date_converter, axis = 1)
    df_transactions["week"] = df_transactions.apply(week_of_year, axis = 1)
    return df_transactions

def select_product_from_inventory(product, inventory):
    return inventory[product]

def select_product_from_transactions(product, transactions):
    return transactions[transactions['description'] == product]


