import requests

city = input("Enter the city name: ")

url = f"http://api.weatherapi.com/v1/current.json?key=00e7c0e0af4e45aca2244140260809&q={city}&aqi=no"

payload = {}
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
