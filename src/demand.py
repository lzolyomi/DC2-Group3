import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_data(filtered_transactions, discounts_per_day=False):
    """
    The function accepts a filtered version of the transactions dataframe,
    and prepares it for a demand function
    OPTIONAL: supply the discounts_per_day dataframe for added predictor
    """

    #converts day to datetime
    filtered_transactions["day"] = pd.to_datetime(filtered_transactions["day"])

    #adds discount column
    filtered_transactions["discount"] = 100-(
        filtered_transactions["purchase_price"]/filtered_transactions["std_sales_price"])*100

    #group by item and day, return summary stats of discount table
    purchases_grouped = filtered_transactions.groupby(["description", "day"]).describe()
    pp = purchases_grouped["purchase_price"]["mean"]
    std_p = purchases_grouped["std_sales_price"]["mean"]
    purchases = purchases_grouped["discount"]
    #add months column
    purchases["month"] = [purchases.index[i][1].month_name() for i in range(purchases.shape[0])]
    #add product column containing the name of each item
    purchases["product"] = [purchases.index[i][0] for i in range(purchases.shape[0])]
    #add day of year (for plotting reasons)
    purchases["day"] = [purchases.index[i][1].dayofyear for i in range(purchases.shape[0])]
    #add day of week for predictor
    purchases["dayofweek"] = [purchases.index[i][1].dayofweek for i in range(purchases.shape[0])]
    purchases["dayofweek"].replace({0:"Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday"}, inplace=True)
    purchases["purchase_price"] = pp
    purchases["std_sales_price"] = std_p
    purchases.rename(columns={"mean":"discount"}, inplace=True)
    #add square of discount as extra predictor
    purchases["discount_2"] = purchases["discount"]**2
    purchases["prev_day_purchases"] = purchases["count"].shift(1)
    purchases["prev_day_purchases"].iloc[0] = 0

    if type(discounts_per_day) != pd.core.frame.DataFrame:
        return purchases
    else:
        full_purchases = purchases.join(discounts_per_day.set_index("day"))
        return full_purchases

    # TODO:other features: standard price, purchase price
    #take logarithm of price and log of count 
    # run it for one product only
    # find a t-test for coefficients



def fit_ohc(pd_column):
    #from a one column dataframe returns a one hot encoded transformed dataframe, plus the encoder
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(pd_column)
    df = pd.DataFrame(encoder.transform(pd_column), columns=encoder.get_feature_names())
    return df, encoder

def prepare_demand_function(df_prepared):
    """
    fits a linear regression as a demand function.
    Used predictors are: product, discount, discount^2, day of week, month
    returns: dictionary with train X, y test X, y and the encoders used for 
    product, weekday and months one hot encoding
    """
    #columns used as predictors
    pred_columns = ["discount", "discount_2", "purchase_price", "prev_day_purchases", "on_discount"]
    target_col = "count"
    #encoding products
    # TODO: see if works with only one product
    unique_products = df_prepared["product"].unique()
    if len(unique_products) > 1:
        df_products, enc_product = fit_ohc(df_prepared[["product"]])
    else: 
        df_products = pd.DataFrame()
        enc_product = None
    #encoding weekdays
    df_weekday, enc_weekday = fit_ohc(df_prepared[["dayofweek"]])    
    #encoding months
    df_months, enc_months = fit_ohc(df_prepared[["month"]])

    full_df = pd.concat([pd.DataFrame(df_prepared.values, columns=df_prepared.columns), 
    df_products, df_weekday, df_months], axis=1)
    # index at which we want to split (20%split)
    full_df.dropna(inplace=True)
    split_index = full_df.shape[0] #- round(full_df.shape[0]/5)
    train, test = full_df[:split_index], full_df[split_index:]
    #separating X and y for train and test set
    X_train = pd.concat([train[pred_columns], train.filter(regex="x0_")], axis=1)
    y_train = train[target_col].values
    
    X_test = pd.concat([test[pred_columns], test.filter(regex="x0_")], axis=1)
    y_test = test[target_col].values

    summary_dct = {"train":[X_train, y_train], "test":[X_test, y_test],"product_enc":enc_product, "weekday_enc":enc_weekday,"months_enc":enc_months}

    return summary_dct


def fit_demand_function(input_dct, model = Ridge()):
    """
    Fits a Ridge regression as a demand function
    input: a dictionary which at least has train and test containing a list of X and y data
    returns: fitted Ridge regressor object
    """

    X_train, y_train = input_dct["train"][0], input_dct["train"][1]
    X_test, y_test = input_dct["test"][0], input_dct['test'][1]

    linreg = model
    linreg.fit(X_train, y_train)
    try:
        mae = mean_absolute_error(y_test, linreg.predict(X_test))
        mse = mean_squared_error(y_test, linreg.predict(X_test))
        r_score = linreg.score(X_test, y_test)
        print("Model fitted; test set metrics: MAE: {} MSE: {}, R^2 score: {}".format(mae, mse, r_score))
    except(ValueError):
        print("No valuable test data present")

    return linreg

def prepare_predictors(df_prep, input_dct):
    """
    accepts a dataframe prepared by the `prepare_data` function, and a dict with the onehot encoders
    returns a formatted dataframe suitable for predicting with the linear regr
    """
    