import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_resource_ids(url):
    """
    Scrape all resource IDs from the given webpage URL.
    
    Args:
    url (str): The URL of the webpage to scrape.
    
    Returns:
    list: A list of unique resource IDs found on the webpage.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    resource_ids = []

    # Find all 'a' tags with 'href' attributes
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/resource/' in href:
            resource_id = href.split('/resource/')[1].split('/')[0]
            resource_ids.append(resource_id)
    
    # Remove duplicate resource IDs
    resource_ids = list(set(resource_ids))
    
    return resource_ids

def save_resource_ids(resource_ids, output_folder):
    """
    Save the resource IDs to a file.
    
    Args:
    resource_ids (list): List of resource IDs to save.
    output_folder (str): Folder to save the resource IDs file.
    """
    os.makedirs(output_folder, exist_ok=True)
    resource_ids_file = os.path.join(output_folder, 'resource_ids.txt')
    with open(resource_ids_file, 'w') as f:
        for resource_id in resource_ids:
            f.write(f"{resource_id}\n")

def fetch_data(resource_id):
    """
    Fetch data from a resource link and return as a DataFrame.
    
    Args:
    resource_id (str): Resource ID to fetch data for.
    
    Returns:
    DataFrame: Data fetched from the resource.
    str: API URL used for fetching the data.
    """
    api_url = f'https://birmingham-city-observatory.datopian.com/api/3/action/datastore_search?resource_id={resource_id}&limit=999999999'
    response = requests.get(api_url)
    data = response.json()
    
    if data['success']:
        records = data['result']['records']
        df = pd.DataFrame(records)
        # Drop the '_id' column
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
        return df, api_url
    else:
        print(f"Failed to fetch data for resource ID {resource_id}: {data['error']['message']}")
        return pd.DataFrame(), api_url

def fetch_and_save_data(resource_ids, output_folder):
    """
    Fetch data for all resource IDs and save to CSV and pickle files.
    
    Args:
    resource_ids (list): List of resource IDs to fetch data for.
    output_folder (str): Folder to save the data files.
    """
    all_data = pd.DataFrame()
    os.makedirs(output_folder, exist_ok=True)

    # Fetch and concatenate data for each resource ID
    for resource_id in resource_ids:
        df, api_url = fetch_data(resource_id)
        if not df.empty:
            all_data = pd.concat([all_data, df], ignore_index=True)
            print(f"Data fetched successfully from {api_url}")

    # Check for duplicates ignoring the '_id' column
    duplicates = all_data[all_data.duplicated(keep=False)]

    # Save duplicates to a file
    duplicates_csv = os.path.join(output_folder, 'duplicates.csv')
    duplicates.to_csv(duplicates_csv, index=False)

    # Remove duplicates from the DataFrame
    cleaned_data = all_data.drop_duplicates()

    # Save the cleaned data to a CSV file
    output_csv = os.path.join(output_folder, 'data.csv')
    cleaned_data.to_csv(output_csv, index=False)

    # Save the cleaned data to a pickle file
    output_pickle = os.path.join(output_folder, 'data.pkl')
    cleaned_data.to_pickle(output_pickle)

    # Print confirmation
    print(f"Total records fetched: {all_data.shape[0]}")
    print(f"Total duplicates found: {duplicates.shape[0]}")
    print(f"Total records after removing duplicates: {cleaned_data.shape[0]}")
    print(f"Data saved to '{output_csv}' and '{output_pickle}'. Duplicates saved to '{duplicates_csv}'.")

