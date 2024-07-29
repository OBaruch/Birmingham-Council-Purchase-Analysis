import os
import pandas as pd
import data_extraction

# Define the URL of the webpage to scrape
webpage_url = 'https://birmingham-city-observatory.datopian.com/dataset/purchase-card-transactions'
output_folder = 'Data'

# Scrape resource IDs
resource_ids = data_extraction.scrape_resource_ids(webpage_url)

# Save resource IDs to a file
data_extraction.save_resource_ids(resource_ids, output_folder)

# Fetch data for all resource IDs and save to CSV and pickle files
data_extraction.fetch_and_save_data(resource_ids, output_folder)

# Load the cleaned data from the pickle file
cleaned_data_pickle = os.path.join(output_folder, 'data.pkl')
cleaned_data = pd.read_pickle(cleaned_data_pickle)

# Print the loaded data
print(cleaned_data.head())

# Print confirmation
total_records_fetched = cleaned_data.shape[0]
duplicates_found = pd.read_csv(os.path.join(output_folder, 'duplicates.csv')).shape[0]
total_records_after_removing_duplicates = cleaned_data.shape[0]

print(f"Total records fetched: {total_records_fetched}")
print(f"Total duplicates found: {duplicates_found}")
print(f"Total records after removing duplicates: {total_records_after_removing_duplicates}")
print(f"Data saved to '{output_folder}/data.csv' and '{output_folder}/data.pkl'. Duplicates saved to '{output_folder}/duplicates.csv'.")
