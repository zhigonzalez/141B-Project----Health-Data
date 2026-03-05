from CDC_scraping import get_cdc_places
from census_scrape import get_census_data
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

def joined_dataset():
    cdc_df = get_cdc_places()
    census_df = get_census_data()

    merged = pd.merge(cdc_df, census_df, on = "fips", how = "inner")
    merged.to_csv("diabetes_heart_factors.csv", index = False)

    return merged



if __name__ == "__main__":
    df = joined_dataset()



'''
corr = df[["DIABETES","CHD","OBESITY","CSMOKING","LPA","median_household_income",
       "pct_uninsured","pct_bachelors","pct_black","median_age"]].corr()
sns.heatmap(corr, annot = True, cmap = "coolwarm")
plt.show()
'''