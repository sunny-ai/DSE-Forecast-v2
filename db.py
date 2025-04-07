import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import datetime

db_filename = 'dse.db'

conn = sqlite3.connect(db_filename)
c = conn.cursor()

c.execute('''
    CREATE TABLE IF NOT EXISTS dse_data (
        sl TEXT,
        trading_code TEXT,
        ltp TEXT,
        high TEXT,
        low TEXT,
        closep TEXT,
        ycp TEXT,
        change TEXT,
        trade TEXT,
        value_mn TEXT,
        volume TEXT
    )
''')
conn.commit()

today = datetime.date.today().strftime('%Y-%m-%d')
url = f"https://www.dsebd.org/dse_close_price_archive.php?startDate={today}&endDate={today}&inst=All%20Instrument&archive=data"

response = requests.get(url)
response.raise_for_status()  # Raise an error for bad status codes

soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', class_='table-bordered')

headers = [header.get_text(strip=True) for header in table.find_all('th')]

rows = []
for row in table.find_all('tr')[1:]:  # Skip the header row
    cells = row.find_all('td')
    row_data = [cell.get_text(strip=True) for cell in cells]
    rows.append(row_data)

df = pd.DataFrame(rows, columns=headers)

df.to_sql('dse_data', conn, if_exists='replace', index=False)

conn.close()

print(f"Data has been successfully scraped and saved to '{db_filename}'.")
