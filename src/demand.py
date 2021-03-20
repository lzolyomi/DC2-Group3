import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

def prepare_demand_function(filtered_transactions):
    """
    The function accepts a filtered version of the transactions dataframe,
    and prepares it for a demand function
    """
    return_dct = dict() #dictionary for return values
    #converts day to datetime
    filtered_transactions["day"] = pd.to_datetime(filtered_transactions["day"])

    #adds discount column
    filtered_transactions["discount"] = 100-(
        filtered_transactions["purchase_price"]/filtered_transactions["std_sales_price"])*100
    return_dct["items"] = filtered_transactions["description"].unique()

    #group by item and day, return summary stats of discount table
    purchases = filtered_transactions.groupby(["description", "day"]).describe()["discount"]
    #add months column
    purchases["month"] = [purchases.index[i][1].month_name() for i in range(purchases.shape[0])]
    #add product column containing the name of each item
    purchases["product"] = [purchases.index[i][0] for i in range(purchases.shape[0])]
    #add day of year (for plotting reasons)
    purchases["day"] = [purchases.index[i][1].dayofyear for i in range(purchases.shape[0])]
    #add day of week for predictor
    purchases["dayofweek"] = [purchases.index[i][1].dayofweek for i in range(purchases.shape[0])]
    purchases["dayofweek"].replace({0:"Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday"}, inplace=True)
    purchases.rename(columns={"mean":"discount"}, inplace=True)
    #add square of discount as extra predictor
    purchases["discount_2"] = purchases["discount"]**2

    return purchases


def fit_ohc(pd_column):
    #from a one column dataframe returns a one hot encoded transformed dataframe, plus the encoder
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(pd_column)
    df = pd.DataFrame(encoder.transform(pd_column), columns=encoder.get_feature_names())
    return df, encoder

def fit_demand_function(df_prepared):
    """
    fits a linear regression as a demand function.
    Used predictors are: product, discount, discount^2, day of week, month
    returns: TODO
    """
    #columns used as predictors
    pred_columns = ["discount", "discount_2"]
    target_col = "count"
    #encoding products
    df_products, enc_product = fit_ohc(df_prepared[["product"]])
    #encoding weekdays
    df_weekday, enc_weekday = fit_ohc(df_prepared[["dayofweek"]])    
    #encoding months
    df_months, enc_months = fit_ohc(df_prepared[["month"]])

    full_df = pd.concat([pd.DataFrame(df_prepared.values, columns=df_prepared.columns), 
    df_products, df_weekday, df_months], axis=1)
    # index at which we want to split (20%split)
    full_df.dropna(inplace=True)
    split_index = full_df.shape[0] - round(full_df.shape[0]/5)
    train, test = full_df[:split_index], full_df[split_index:]
    #separating X and y for train and test set
    X_train = pd.concat([train[pred_columns], train.filter(regex="x0_")], axis=1)
    y_train = train[target_col].values
    
    X_test = pd.concat([test[pred_columns], test.filter(regex="x0_")], axis=1)
    y_test = test[target_col].values

    summary_dct = {"train":[X_train, y_train], "test":[X_test, y_test],
     "product_enc":enc_product, "weekday_enc":enc_weekday,"months_enc":enc_months}

    return summary_dct


transactions = pd.read_csv(r"C:\Users\zolyo\OneDrive\Documents\Quartile 3\JBG050 Data Challenge 2\Code\src\data\transactions.csv")
filt_transactions = transactions[(transactions["category"] == "vegetable") & (transactions["bio"] == 1)]

test = prepare_demand_function(filt_transactions)
print(test.head())
