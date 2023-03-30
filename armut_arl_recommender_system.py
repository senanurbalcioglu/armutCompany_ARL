####################
# BUSINESS PROBLEM
####################

# Armut, Turkey's largest online service platform, brings together service providers and those who want to receive service.
# It provides easy access to services such as cleaning, modification and transportation via computer or smart phone.
# It is desired to create a product recommendation system with Association Rule Learning by using the
# data set containing the service users and the services and categories they have received.


###################
# DATASET
###################
# The data set consists of the services received by the customers and the categories of these services.
# It contains the date and time information of each service received.

# UserId: Customer Number
# ServiceId: Anonymized services belonging to each category. (Example: Upholstery washing service under the cleaning category)
# A ServiceId can be found under different categories and refers to different services under different categories.
# (Example: Service with CategoryId 7 and ServiceId 4 is honeycomb cleaning, while service with CategoryId 2 and ServiceId 4 is furniture assembly)
# CategoryId: Anonymized categories. (Example: Cleaning, transportation, renovation category)# CreateDate: The date the service was purchased


#########################
# TASK 1: PREPARING THE DATA
#########################

# Step 1: Read the armut_data.csv file

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df_ = pd.read_csv(r"datasets/armut_data.csv")
df = df_.copy()


def check_df(dataframe, head=5):
    print("INFO".center(70,'='))
    print(dataframe.info())

    print("SHAPE".center(70,'='))
    print('Rows: {}'.format(dataframe.shape[0]))
    print('Columns: {}'.format(dataframe.shape[1]))

    print("TYPES".center(70,'='))
    print(dataframe.dtypes)

    print("HEAD".center(70, '='))
    print(dataframe.head(head))

    print("TAIL".center(70,'='))
    print(dataframe.tail(head))

    print("NULL".center(70,'='))
    print(dataframe.isnull().sum())

    print("QUANTILES".center(70,'='))
    print(dataframe.describe().T)

check_df(df)

# Step 2: ServiceID represents a different service for each CategoryID.
# Combine ServiceID and CategoryID with "_" to create a new variable to represent the services.
# Step 3: The data set consists of the date and time the services are received, there is no basket definition (invoice, etc.).
# In order to apply Association Rule Learning, a basket (invoice, etc.) definition must be created.
# Here, the definition of basket is the services that each customer receives monthly.
# For example; A basket of 9_4, 46_4 services received by the customer with id 7256 in the 8th month of 2017;
# The 9_4, 38_4 services received in the 10th month of 2017 represent another basket.
# Baskets must be identified with a unique ID. To do this, first create a new date variable containing only the year and month. Combine UserID and the newly created date variable with "_" and assign it to a new variable called ID.

def prepare_dataset(dataframe):
    dataframe["Service"] = dataframe[["ServiceId", "CategoryId"]].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    dataframe["New_Date"] = pd.to_datetime(dataframe["CreateDate"]).apply(lambda x: x.strftime('%Y-%m'))
    dataframe["BasketID"] = ["_".join(x.astype(str)) for x in dataframe[["UserId", "New_Date"]].values]
    return dataframe


df = prepare_dataset(df)


#########################
# TASK 2: Create Association Rules
#########################

# Step 1: Create the basket service pivot table as follows.

# Service         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..


def basket_service_df(dataframe):
    dataframe = dataframe.groupby(["BasketID", "Service"]).agg({"Service": "count"}).unstack().fillna(0). \
        applymap(lambda x: 1 if x > 0 else 0)
    dataframe.columns = dataframe.columns.droplevel()
    return dataframe


df_basket = basket_service_df(df)


# Step 2: Create association rules.
def rules(dataframe):
    freq_items = apriori(dataframe, min_support=0.01, use_colnames=True)
    rule = association_rules(freq_items, metric="support", min_threshold=0.01)
    return rule


df_rule = rules(df_basket)


# Step 3: Using the arl_recommender function, recommend a service to a user who last received the 2_0 service.
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    sorted_rules = sorted_rules[sorted_rules["antecedents"].apply(lambda x: product_id in x)]
    recommendation_list = [i for i in sorted_rules["consequents"].apply(list)]
    return recommendation_list[:rec_count]


rec = arl_recommender(df_rule, "2_0")
print(f" Recommendation List: {rec}")
