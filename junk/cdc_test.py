import requests
import pandas as pd

'''
disregard this, just needed to test something earlier where my url wasn't working
'''


def get_cdc_places():
    '''
    wrong api id --> test a few
    '''
    dataset_ids = [
        "cwsq-ngmh",  # PLACES 2023
        "duw2-7jbt",  # PLACES county data
        "swc5-untb",  # old fallback
    ]
    
    for dataset_id in dataset_ids:
        url = f"https://data.cdc.gov/resource/{dataset_id}.json"
        params = {"$limit": 3}
        resp = requests.get(url, params=params)
        print(f"\n {dataset_id}, status: {resp.status_code}")
        
        result = resp.json()
        if isinstance(result, list) and len(result) > 0:
            print("good : columns:", list(result[0].keys()))
            break
        else:
            print("bad : ", result)

get_cdc_places()
