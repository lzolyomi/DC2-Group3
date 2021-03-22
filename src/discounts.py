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