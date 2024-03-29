import pandas as pd 
import numpy as np

def disc_per_day(transactions):
    """
    from the transactions dataframe, returns a new dataframe containing the nr
    of discounted items on that day 
    """

    transactions["on_discount"] = np.where(transactions["std_sales_price"] == transactions["purchase_price"], 0, 1)
    grouped_transactions = transactions.groupby(["description", "day"]).describe()
    grouped_transactions["day"] = [grouped_transactions.index[i][1] for i in range(grouped_transactions.shape[0])]

    df_discounts = pd.DataFrame({"days":[grouped_transactions.index[i][1] for i in range(grouped_transactions.shape[0])], 
    "on_discount":grouped_transactions["on_discount"]["count"]})
    discount_per_day = df_discounts.groupby("days").sum()
    discount_per_day["day"] = pd.to_datetime(discount_per_day.index, dayfirst=True)

    return discount_per_day

def apply_discount(input_predictors, applicable_discount):
    """
    Accepts a dataframe formatted for prediction and applies (further) discounts to it
    before prediction
    """
    predictors = input_predictors.copy()
    if sum(predictors["discount"]) < 1:
        #when there is no already existing discount
        predictors["discount"] = applicable_discount
        #predictors["discount_2"] = applicable_discount**2
        predictors["purchase_price"] = predictors["purchase_price"]*(1-applicable_discount/100)
    else:
        predictors["purchase_price"] = predictors["purchase_price"]/(1-predictors["discount"]/100) #get original purchase price

        predictors["discount"] = predictors["discount"] + applicable_discount
        #predictors["discount_2"] = (predictors["discount_2"]**0.5 + applicable_discount)**2
        predictors["purchase_price"] = predictors["purchase_price"]*predictors["purchase_price"]*(1-predictors["discount"]/100)
    predictors["on_discount"] = predictors["on_discount"]+1
    return predictors

def discount_bins(sold_ratio):
    """
    For a given sold_ratio, returns a tuple of discount levels, predetermined by the ratio
    NOTE: it is possible to modify discounts and split points freely to see the changes
    """
    if sold_ratio < 0.1:
        discounts = (25,40)
    elif sold_ratio < 0.2:
        discounts = (20,30)
    elif sold_ratio < 0.4:
        discounts = (20, 20)
    elif sold_ratio < 0.6:
        discounts = (10,20)
    else:
        discounts = (0,10)
    return discounts