# FILE: Dashboard/weather_fetch.py
# Weather API integration using WeatherAPI.com

import requests
from datetime import datetime

API_KEY = "00e7c0e0af4e45aca2244140260809"  


def fetch_weather_forecast(city: str, target_date: datetime) -> dict | None:
    """
    Fetch weather forecast for a given city and date.
    NOTE: WeatherAPI.com free tier only gives current weather + 3-day forecast.
    If target_date is today, current weather is returned.
    If target_date is within next 3 days, forecast is returned.
    Otherwise, returns None.
    """
    try:
        # Check if target_date is today or in future (max 3 days)
        today = datetime.now().date()
        target = target_date.date()
        days_diff = (target - today).days

        if days_diff < 0:
            print(f"  Cannot fetch past weather for {target_date}")
            return None

        if days_diff == 0:
            # Today → use current weather endpoint
            url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature": data["current"]["temp_c"],
                    "humidity": data["current"]["humidity"],
                    "windspeed": data["current"]["wind_kph"],
                    "precipitation": data["current"]["precip_mm"],
                    "rainfall": "Yes" if data["current"]["precip_mm"] > 0 else "No",
                }
            else:
                print(f"  API error: {response.status_code}")
                return None

        elif 1 <= days_diff <= 3:
            # Forecast for next 3 days
            url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={city}&days=3&aqi=no"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # Find forecast for target date
                for day in data["forecast"]["forecastday"]:
                    if day["date"] == target.strftime("%Y-%m-%d"):
                        return {
                            "temperature": day["day"]["avgtemp_c"],
                            "humidity": day["day"]["avghumidity"],
                            "windspeed": day["day"]["maxwind_kph"],
                            "precipitation": day["day"]["totalprecip_mm"],
                            "rainfall": "Yes" if day["day"]["totalprecip_mm"] > 0 else "No",
                        }
                print(f"  No forecast found for {target_date}")
                return None
            else:
                print(f"  API error: {response.status_code}")
                return None

        else:
            print(f"  WeatherAPI.com free tier only supports 3-day forecast. Date {target_date} is too far.")
            return None

    except Exception as e:
        print(f"  Error fetching weather: {e}")
        return None


 
# For testing directly (optional)
 
if __name__ == "__main__":
    city = input("Enter city name: ")
    date_str = input("Enter date (YYYY-MM-DD) or press Enter for today: ")
    if date_str:
        target = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        target = datetime.now()

    weather = fetch_weather_forecast(city, target)
    if weather:
        print("\n Weather fetched successfully:")
        print(f"    Temperature: {weather['temperature']}°C")
        print(f"   Humidity: {weather['humidity']}%")
        print(f"   Wind Speed: {weather['windspeed']} km/h")
        print(f"   Precipitation: {weather['precipitation']} mm")
        print(f"    Rainfall: {weather['rainfall']}")
    else:
        print("Could not fetch weather.")



# import requests

# city = input("Enter the city name: ")

# url = f"http://api.weatherapi.com/v1/current.json?key=00e7c0e0af4e45aca2244140260809&q={city}&aqi=no"

# payload = {}
# headers = {}

# response = requests.request("GET", url, headers=headers, data=payload)

# print(response.text)
