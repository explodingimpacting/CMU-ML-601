import json
import requests
import numpy as np
import pandas as pd
import time
import requests
from requests.auth import HTTPBasicAuth
import time
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings

disable_warnings(InsecureRequestWarning)

def search_github_repos(query, max_pages=10, username=None, token=None, min_stars=10):
    repos = []
    url = "https://api.github.com/search/repositories"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'

    for page in range(1, max_pages + 1):
        params = {
            'q': f"{query} stars:>{min_stars}",
            'sort': 'stars',
            'order': 'desc',
            'per_page': 100,
            'page': page
        }

        response = requests.get(url, headers=headers, params=params, auth=(username, token), verify=False)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data:
                repos.extend(data['items'])
            if len(data['items']) < 100:  # If less than 100 items, we are on the last page
                print("reached last page")
                break
        elif response.status_code == 422:
            print(f"Failed to fetch data: {response.status_code} - {response.json().get('message')}")
            break
        elif response.status_code == 403:
            print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            continue
        else:
            print(f"Failed to fetch data: {response.status_code} - {response.content}")
            break
        time.sleep(1)  # To respect rate limits

    return repos

def save_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {filename}")

    ## GITHUB creds
username = ""  # Replace with your GitHub username
token = ""  # Replace with your GitHub token

## Query
search_string = "NERF" ## modify search string here (can include AND, OR, *, and other search logic)
file_path = "/Users/rwhite/github_repos.json"

queries = {
    search_string: file_path
}
min_stars = 1 # Adjust the minimum star count as needed

for query, filename in queries.items():
    print(f"Searching for repositories related to '{query}' with at least {min_stars} stars...")
    data = search_github_repos(query, max_pages=10, username=username, token=token, min_stars=min_stars)
    save_to_json(data, filename)
    print('-' * 100)

    ## Load the JSON file
with open(filename, 'r') as file:
    github_repos = json.load(file)

# Convert each json to a pandas DataFrame
github_json = '/Users/rwhite/github_repos.json'

df = pd.read_json(github_json)
df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime("%m/%d/%Y")
df.head()
df.to_csv('/Users/rwhite/Downloads/LitReview/analyses/github.csv')

