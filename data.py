import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the DSE archive page
url = 'https://www.dsebd.org/day_end_archive.php?startDate=2025-01-01&endDate=2025-03-31&inst=All%20Instrument&archive=data'

# Send a GET request to the URL
response = requests.get(url)
response.raise_for_status()  # Raise an error for bad status codes

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table; adjust the class or other identifiers as necessary
table = soup.find('table', class_='table-bordered')

# Extract headers
headers = [header.get_text(strip=True) for header in table.find_all('th')]

# Extract rows
rows = []
for row in table.find_all('tr')[1:]:  # Skip the header row
    cells = row.find_all('td')
    row_data = [cell.get_text(strip=True) for cell in cells]
    rows.append(row_data)

# Create a DataFrame
df = pd.DataFrame(rows, columns=headers)

# Save to CSV
df.to_csv('dse_data.csv', index=False)

print("Data has been successfully scraped and saved to 'dse_data.csv'.")
