import pandas as pd
import requests

CENSUS_API_KEY = "daede620bf11f9e7e9f820296d6e9315db2250b3"  #don't kill my key pls

def get_census_data():
    #county level, 5 year est
    variables = {
        "B19013_001E": "median_household_income",
        "B15003_022E": "bachelors_degree_count",
        "B01002_001E": "median_age",
        "B27001_001E": "total_health_insurance",
        "B27001_005E": "uninsured_count",
        "B02001_003E": "black_pop",
        "B02001_002E": "white_pop",
        "B01003_001E": "total_pop",  #can add more later for more features
    }

    var_string = ",".join(variables.keys())
    url = f"https://api.census.gov/data/2022/acs/acs5"

    params = {
        "get": f"NAME,{var_string}",
        "for": "county:*",
        "in": "state:*",
        "key": CENSUS_API_KEY
    }

    response = requests.get(url, params = params)
    data = response.json()
    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns = headers)
    df.rename(columns = variables, inplace = True)

    df["fips"] = df["state"] + df["county"]  #need for fips code to match cdc data

    for col in variables.values():
        df[col] = pd.to_numeric(df[col], errors = "coerce")

    df["pct_uninsured"] = df["uninsured_count"] / df["total_pop"] * 100  #convert to features
    df["pct_bachelors"] = df["bachelors_degree_count"] / df["total_pop"] * 100
    df["pct_black"] = df["black_pop"] / df["total_pop"] * 100

    return df

'''
if __name__ == "__main__":
    df = get_census_data()
    print(df.head())
'''