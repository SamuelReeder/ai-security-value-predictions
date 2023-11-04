import yfinance as yf
import requests
import json

url = "https://twelve-data1.p.rapidapi.com/stocks"

exchanges = ["NASDAQ", "NYSE"]
querystring = {"exchange":"NYSE","format":"json"}

headers = {
	"X-RapidAPI-Key": "023d59411cmshb7d84215a410b60p117fc8jsn5ec37e39242c",
	"X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())

# Serializing json
json_object = json.dumps(response.json(), indent=4)
 
# Writing to sample.json
with open("nsye.json", "w") as outfile:
    outfile.write(json_object)