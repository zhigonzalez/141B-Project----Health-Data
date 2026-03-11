import pandas as pd
import requests

def get_cdc_places():
    url = "https://data.cdc.gov/resource/cwsq-ngmh.json" #no key -- free

    measures = ["DIABETES", "CHD", "OBESITY", "CSMOKING", "BINGE", "LPA", "SLEEP", "MHLTH", "ACCESS2"] #add more/less later
    #risk_factors = ["OBESITY", "CSMOKING", "BINGE", "LPA", "SLEEP"]
    #all_measures = measures + risk_factors

    measure_filter = " OR ".join([f"measureid='{m}'" for m in measures])

    params = {
        "$where": f"({measure_filter}) AND countyfips IS NOT NULL",
        "$select": "countyname, countyfips, stateabbr, measureid, data_value",
        "$limit": 600000  #change later for more data --> 1000 for testing --> 600000 gets all US counties
    }

    response = requests.get(url, params = params)
    df = pd.DataFrame(response.json())
    
    '''
    print("columns:", df.columns.tolist())
    print("some rows:")
    print(df.head())
    print(response.status_code)  #check resp cuz this not working/giving what i want
    '''

    df["data_value"] = pd.to_numeric(df["data_value"], errors = "coerce")
    #each measure --> column
    df_pivot = df.pivot_table(
        index = ["countyfips", "countyname", "stateabbr"],
        columns = "measureid",
        values = "data_value",
        aggfunc = "first"
    ).reset_index()

    df_pivot.columns.name = None
    df_pivot.rename(columns = {"countyfips": "fips"}, inplace = True)


    return df_pivot


'''
if __name__ == "__main__":
    df = get_cdc_places()
    print(df.head())
'''