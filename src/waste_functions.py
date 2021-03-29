import numpy as np
import pandas as pd 
import datetime
from demand import prepare_predictors
from discounts import apply_discount
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
    df_waste["best before day"] = df_waste["best before day"]
    return df_waste

def add_purchases(df_waste, transactions, product):
    """
    Extends the dataframe created by the previous function with purchase numbers

    """
    filtered_transactions = transactions[transactions["description"] == product]
    def input_DOY(row):
        doy = pd.to_datetime(row["day"], dayfirst=True).dayofyear - 1
        return doy 
    filtered_transactions["DOY"] = filtered_transactions.apply(input_DOY, axis=1)
    cumsum_purchases = filtered_transactions.groupby("DOY").count()["product_id"].cumsum()
    #creating a dataframe,  merging it with reference so each day is accounted for
    df_purchases = pd.DataFrame({"DOY":cumsum_purchases.index, "cumulative purchases":cumsum_purchases})
    ref = pd.DataFrame({"doy":range(1,365)}) #reference table to match up days
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

def get_ranges(inventory, product):
    """
    for a given product returns a list of tuples, containing the ranges of best before dates
    """
    best_before = []
    for element_list in inventory[product]:
        for batch in element_list:
            best_before.append(batch[0])
    bb_dates = list(np.unique(best_before))
    #list of each unique best before dates
    bb_dates.insert(0, 0)
    ranges = []
    i = 0
    for item in bb_dates:
        if i < len(bb_dates)-1:
            i += 1
            ret_tupl = (item, bb_dates[i])
            ranges.append(ret_tupl)
    return ranges

def predicted_demand(df_waste, ranges, model, input_dct, prep_transactions):
    """
    Adds the predicted demand for the df_waste returned by the add purchases function
    """
    pred_values = []
    opt_values = []
    for rang in ranges:
        #TODO: check if there is actual waste and only apply discounts then
        df_frange = prep_transactions[(prep_transactions["DOY"] >= rang[0]) & (prep_transactions["DOY"] <= rang[1])]
        try:
            frange_prepared = prepare_predictors(df_frange, input_dct)
            #TODO: add discounts 
            frange_mod = frange_prepared.copy() #copy of the prepared df for appliying discounts
            last = len(frange_mod)-1
            frange_mod.loc[last:last] = apply_discount(frange_mod.loc[last:last],30)
            frange_mod.loc[last-1:last-1] = apply_discount(frange_mod.loc[last-1:last-1],20)
            est_demand = model.predict(frange_prepared).sum()
            est_mod_demand = model.predict(frange_mod).sum()
            pred_values.append(est_demand)
            opt_values.append(est_mod_demand)
        except(ValueError):
            pass
    df_waste["predicted demand"] = pred_values
    df_waste["demand_discounts"] = opt_values
    return df_waste
