import numpy as np
import pandas as pd 
import datetime
import ast 
from sklearn.linear_model import Ridge
import statistics as stat

from demand import *
from discounts import *

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
    def total_inv(row):
        total = 0
        if len(row[product]) > 0:
            for batch in row[product]:
                total += batch[1]
        return total
    filter_inventory["total inventory"] = filter_inventory.apply(total_inv, axis = 1)

    df_waste['total_inventory'] = [int(filter_inventory.loc[filter_inventory[(filter_inventory['before or after delivery'] == 'before') & (filter_inventory['day'] == day)].index[0],
     'total inventory']) if len(filter_inventory[(filter_inventory['before or after delivery'] == 'before') & (filter_inventory['day'] == day)].index) > 0
      else None for day in df_waste['best before day']]

    df_waste['remaining_stock'] = df_waste['total_inventory'] - df_waste['amount']
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
            if bb_dates[i] < 365: #we don't have any purchases  when year is 366 long
                ranges.append(ret_tupl)
    return ranges

def predicted_demand(df_waste, ranges, model, input_dct, prep_transactions):
    """
    Adds the predicted demand for the df_waste returned by the add purchases function
    """
    pred_values = []
    opt_values = []
    avg_salesprice = []
    std_price = []
    estimated_ratios = []
    i = 0
    for rang in ranges:
        #TODO: check if there is actual waste and only apply discounts then
        df_frange = prep_transactions[(prep_transactions["DOY"] >= rang[0]) & (prep_transactions["DOY"] <= rang[1])]
        purchases = df_frange["count"].sum()
        prices = (df_frange["count"]*df_frange["purchase_price"]).sum()
        try:
            frange_prepared = prepare_predictors(df_frange, input_dct)
            #TODO: add discounts 
            frange_mod = frange_prepared.copy() #copy of the prepared df for appliying discounts
            last = len(frange_mod)-1
            est_demand = model.predict(frange_prepared).sum()
            # >>>> used to determine whether we expecct any waste
            last = len(df_frange)-1 #index of last element, i.e. day it goes bad
            if last > 1:#only if range has at least two days where we can apply discounts
                #items purchases from batch 2days before going bad
                purchases_before_disc = pd.DataFrame(df_frange.values, columns=df_frange.columns).loc[:last-2]["count"].sum()
                predictions_before_disc = model.predict(frange_prepared.loc[:last-2]).sum()
                #prediction of demand for the last two days
                pred_last_days = model.predict((frange_prepared.loc[last-1:])*(purchases_before_disc/predictions_before_disc)).sum()
                #print("Ratio between actual purchases/estimated: {}".format(purchases_before_disc/predictions_before_disc))
                estimated_ratios.append(purchases_before_disc/predictions_before_disc)
                #purchases in the last two days used to determine inventory before the days
                actual_last_days = pd.DataFrame(df_frange.values, columns=df_frange.columns).loc[last-1:]["count"].sum()
                remaining_inv = df_waste.iloc[i]["remaining"].sum() + actual_last_days #products remain 2days before going bad
                #TODO: predict values before 2 days, calculate ratio
                if (remaining_inv - pred_last_days) > 0: #if we have expected waste waste
                    frange_mod = frange_prepared.copy() #copy of the prepared df for appliying discounts
                    batch_size = df_waste.iloc[i]["amount"] #size of shipment received
                    sold_ratio = purchases_before_disc/batch_size
                    disc_levels = discount_bins(sold_ratio)
                    frange_mod.loc[last-1:last-1] = apply_discount(frange_mod.loc[last-1:last-1],disc_levels[0]) #day before goes bad
                    frange_mod.loc[last:last] = apply_discount(frange_mod.loc[last:last],disc_levels[1]) #actual day it goes bad
                    #print("Ratio of sold/shipment is {}".format(sold_ratio))
                    est_mod_demand = model.predict(frange_mod).sum()
                    ratio = float(est_mod_demand/est_demand)
                    
                    sales_prices = (frange_mod["purchase_price"].values*(df_frange["count"].values*ratio)).sum()
                    purchases = df_frange["count"].sum()*ratio
                    avg = sales_prices/purchases
                    opt_values.append(round(est_mod_demand))
                    
                    avg_salesprice.append(avg)
                else:
                    opt_values.append(round(est_demand))
                    avg_salesprice.append(prices/purchases)
                    
            else:
                opt_values.append(round(est_demand))
                avg_salesprice.append(prices/purchases)
            pred_values.append(round(est_demand))
            i += 1
            std_price.append(frange_prepared["purchase_price"].mean())
            
        except(ValueError):
            if len(opt_values) < df_waste.shape[0]-1: #if range does not have any purchases from the given product
                opt_values.append(0)
                avg_salesprice.append(0)
                std_price.append(0)

            print("Value error occured at range {}".format(rang))
            pass
        
    df_waste["predicted demand"] = pred_values
    df_waste["demand_discounts"] = opt_values
    ratio=sum(opt_values)/sum(pred_values)
    df_waste["avg_price"] = avg_salesprice
    df_waste["std_price"] = std_price
    df_waste["expected purchases"] = round(df_waste["purchases"]*ratio)
    df_waste["expected waste"] = df_waste["waste"] - (df_waste["expected purchases"]-df_waste["purchases"]) 
    #TODO: if negative excess amount CHECK
    df_waste["expected waste"] = df_waste[df_waste["expected waste"] > 0]["expected waste"]
    print("Average of actual/estimated purchase ratios: {}".format(stat.mean(estimated_ratios)))
    return df_waste


def waste_analysis(inventory, transactions, df_product, product:str, model = Ridge(), discount_per_day=None, compl_prod=False):
    """
    Contains every function that is needed to return the analysis in the combininb_waste notebook
    """
    #convert string into tuples in the inventory table
    for colname in inventory.columns[2:]:
        inventory[colname] = [ast.literal_eval(i) for i in inventory[colname]]

    filtered_transactions = transactions[transactions["description"] == product]
    filtered_inventory = inventory[["day", "before or after delivery", product]]
    waste_df = create_bb_data(filtered_inventory, product)
    waste_df_ext = add_purchases(waste_df, transactions, product)
    print("Extended waste dataframe created")
    ranges = get_ranges(inventory, product)
    if type(discount_per_day) == pd.core.frame.DataFrame:
        discounts_per_day = discount_per_day.copy()
    else:
        print("No daily discount table supplied, calculating...")
        discounts_per_day = disc_per_day(transactions)
    #TODO: add fix to complimentary product
    prepared_transactions = prepare_data(filtered_transactions, discounts_per_day=discounts_per_day, complimentary_products=compl_prod, full_transactions=transactions)
    print("Data prepared for prediction")
    output_dct = prepare_demand_function(prepared_transactions)
    model = fit_demand_function(output_dct, model)
    print("Demand function fitted")
    test = predicted_demand(waste_df_ext, ranges, model, output_dct, prepared_transactions)
    print("Waste change predicted, calculating revenue loss...")
    purchase_price = df_product[df_product["description"] == product]["purchase_price"].values
    waste_df_ext.fillna({"expected waste":0}, inplace=True)
    waste_df_ext["expected loss revenue"] = (waste_df_ext["expected waste"]*waste_df_ext["avg_price"]).round(2)
    waste_df_ext["expected loss profit"] = waste_df_ext["expected waste"]*(waste_df_ext["avg_price"]-purchase_price)
    waste_df_ext["expected waste cost"] = waste_df_ext["expected waste"]*purchase_price
    #TODO: fix average price
    waste_df_ext["loss revenue"] = waste_df_ext["waste nn"]*waste_df_ext["std_price"]
    waste_df_ext["loss profit"] = waste_df_ext["waste nn"]*(waste_df_ext["std_price"]-purchase_price)
    waste_df_ext["waste cost"] = waste_df_ext["waste nn"]*purchase_price

    return waste_df_ext