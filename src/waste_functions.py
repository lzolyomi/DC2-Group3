import numpy as np
import pandas as pd 
import datetime
def create_bb_data(inventory, product):
    """
    For a given product it returns a dataframe containing extensive info about wasted items
    """
    filter_inventory = inventory[["day","before or after delivery", product]]

    best_before = []
    for element_list in filter_inventory[product]:
        for batch in element_list:
            best_before.append(batch[0])
    bb_dates = list(np.unique(best_before))

    df_waste = pd.DataFrame({"best before day":bb_dates})
    df_waste = df_waste[df_waste["best before day"] < 365]
    def input_inv(row):
        for element_list in filter_inventory[product]:
            if len(element_list) > 0:
                for batch in element_list:
                    if batch[0] == row["best before day"]:
                        return batch[1]
    df_waste["amount"] = df_waste.apply(input_inv, axis=1)
    #TODO: add purchases, interval and cumulative

    return df_waste

def add_purchases(df_waste, transactions, product):
    """
    Extends the dataframe created by the previous function with purchase numbers

    """
    filtered_transactions = transactions[transactions["description"] == product]
    def input_DOY(row):
        doy = pd.to_datetime(row["day"]).dayofyear
        return doy 
    filtered_transactions["DOY"] = filtered_transactions.apply(input_DOY, axis=1)
    cumsum_purchases = filtered_transactions.groupby("DOY").count()["product_id"].cumsum()
    #creating a dataframe,  merging it with reference so each day is accounted for
    df_purchases = pd.DataFrame({"DOY":cumsum_purchases.index, "cumulative purchases":cumsum_purchases})
    ref = pd.DataFrame({"doy":range(365)}) #reference table to match up days
    joint_purchases = ref.set_index('doy').join(df_purchases)
    joint_purchases.fillna(method='pad', inplace=True)#days where 0 purchases happened
    waste_df = df_waste.set_index('best before day').join(joint_purchases)
    waste_df['purchases'] = waste_df["cumulative purchases"].diff()
    waste_df["remaining"] = waste_df["amount"] - waste_df['purchases']
    waste_df["remaining"].iloc[0] = waste_df["amount"].iloc[0]-waste_df["cumulative purchases"].iloc[0]
    print(waste_df.shape)
    length = waste_df.shape[0]
    lst_waste = []
    for i in range(length):
        #iterates through the rows of the waste dataframe
        current_row = waste_df.iloc[i]
        if i == 0:
            #
            lst_waste.append(current_row["remaining"])
            continue     
        else:
            prev_row = waste_df.iloc[i-1] #previous row in dataframe
            if lst_waste[-1] < 0:
                #no idea whats happening here
                waste = lst_waste[-1] + current_row["remaining"]
                lst_waste.append(lst_waste[-1] + current_row["remaining"])
            else:
                lst_waste.append(current_row["remaining"])
        
    def input_weeknr(row):
        date=datetime.datetime(2018,1,1)+datetime.timedelta(row["DOY"])
        week = date.isocalendar()[1]
        return week

    waste_df["waste"] = lst_waste
    waste_df["week"] = waste_df.apply(input_weeknr, axis=1)
    waste_df["waste nn"] = [i if i> 0 else 0 for i in waste_df['waste']]

    return waste_df
