import json
import requests
import csv
import os

with open('nasdaq.json') as f:
    nasdaq_data = json.load(f)

with open('nyse.json') as f:
    nyse_data = json.load(f)

stock_symbols = [stock['symbol'] for stock in nasdaq_data['data']] + [stock['symbol'] for stock in nyse_data['data']]

rapidapi_key = os.environ.get('RAPIDAPI_KEY')

url = "https://mboum-finance.p.rapidapi.com/hi/history"
headers = {
    "X-RapidAPI-Key": rapidapi_key,
    "X-RapidAPI-Host": "mboum-finance.p.rapidapi.com"
}

fetched_symbols = set()
missed_symbols = set()

with open('stock_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    
    writer.writerow(['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    for symbol in stock_symbols:
        if symbol in fetched_symbols:
            continue
        
        params = {"symbol": symbol, "interval": "1d", "diffandsplits": "false"}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raises a HTTPError if the response status is 4xx, 5xx
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {symbol}: {e}")
            missed_symbols.add(symbol)
            continue
        
        try:
            for timestamp, point in data['items'].items():
                try:
                    writer.writerow([
                        symbol,  
                        point.get('date', 'N/A'), 
                        point.get('open', 'N/A'), 
                        point.get('high', 'N/A'), 
                        point.get('low', 'N/A'), 
                        point.get('close', 'N/A'), 
                        point.get('volume', 'N/A')
                    ])
                except Exception as e:
                    print(f"Error writing row for symbol {symbol} at timestamp {timestamp}: {e}")
        except KeyError as e:
            print(f"Error fetching data for {symbol}: {e}")
            missed_symbols.add(symbol)
            continue
        
        fetched_symbols.add(symbol)
        if len(fetched_symbols) % 10 == 0:
            print(f'{len(fetched_symbols)} symbols fetched, {len(fetched_symbols) / len(stock_symbols) * 100:.2f}% complete')

print(missed_symbols)
if missed_symbols != set():
    with open("missed.txt", "w") as outfile:
        outfile.write('\n'.join(missed_symbols))