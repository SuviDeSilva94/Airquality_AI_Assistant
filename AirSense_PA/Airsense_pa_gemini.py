import requests
import json
import google.generativeai as genai
import logging
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

# ----- Logging Setup -----
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- API Configuration -----
API_KEY = ""  # OpenWeatherMap API key
GEMINI_API_KEY = ""  # Gemini API key from Google AI Studio
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
GEO_URL = "http://api.openweathermap.org/geo/1.0/direct"

# ----- Default Location Configuration -----
DEFAULT_LATITUDE = 59.3293  # Latitude for Stockholm, Sweden
DEFAULT_LONGITUDE = 18.0686  # Longitude for Stockholm, Sweden


# ----- Gemini Setup -----
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# ----- Flask App Setup -----
app = Flask(__name__)
CORS(app) # Enable CORS for all routes with a single line of code!

# --- Initial Default Values ---
current_latitude = DEFAULT_LATITUDE
current_longitude = DEFAULT_LONGITUDE
current_city_name = "Stockholm"


# ----- Geocoding function -----
def get_coordinates(city_name):
    """
      Fetches coordinates for a given city using OpenWeatherMap geocoding API

       Parameters:
          city_name (str): The name of the city

       Returns:
          tuple or None: A tuple containing the (latitude, longitude), or None if an error occurs.
    """
    url = f"{GEO_URL}?q={city_name}&limit=1&appid={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data:
            coordinates = data[0]['lat'], data[0]['lon']
            print(f"get_coordinates(): Successfully got coordinates: {coordinates} for {city_name}") #Debug message
            return coordinates
        else:
            logging.error(f"No coordinates found for city: {city_name}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching coordinates for city {city_name}: {e}")
        return None


# ----- API Request Function -----
def get_air_quality_data(lat, lon):
    """
       Fetches air quality data from OpenWeatherMap.

        Parameters:
            lat (float): Latitude of the location.
            lon (float): Longitude of the location.

        Returns:
            dict or None: Air quality data as a dictionary, or None if an error occurs.
    """
    url = f"{BASE_URL}?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return process_api_data(data)
    except requests.exceptions.RequestException as e:
       logging.error(f"Error fetching air quality data: {e}")
       return None

# ----- Data Processing Function -----
def process_api_data(data):
    """
        Returns the raw JSON data from the API.

        Parameters:
            data (dict): The raw JSON response from the API.

        Returns:
           dict: The raw JSON dictionary
    """
    if data and 'list' in data and data['list']:
        return data # Return the whole dictionary
    else:
        logging.error(f"Data format not recognised: {data}")
        return None

# ----- Activity Recommendation Logic -----
def get_activity_recommendation(aqi):
    """
       Determines activity recommendations based on Air Quality Index.

        Parameters:
           aqi (int): The Air Quality Index Value

        Returns:
           str:  Recommendation for outdoor activity
    """
    if aqi is None:
      return "Could not determine air quality."
    if aqi <= 1:
        return "Air quality is good. It's a great day for jogging."
    elif aqi <= 2:
        return "Air quality is moderate. You can jog but may consider a shorter time period."
    elif aqi <= 3:
        return "Air quality is unhealthy for sensitive groups. It's best to avoid strenuous outdoor activities like jogging."
    elif aqi <= 4:
        return "Air quality is unhealthy. Avoid outdoor activities."
    elif aqi <= 5:
        return "Air quality is very unhealthy. Please stay indoors."
    elif aqi > 5:
        return "Air quality is hazardous. Please stay indoors."
    else:
        return "Error determining suitable activities."

# ----- Format API Response Function -----
def format_api_response(data):
    """
       Formats the raw API response into a more readable string.
    """
    formatted_string = f"Air Quality Data from {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n"
    formatted_string += f"Coordinates: Lon = {data.get('coord', {}).get('lon')}, Lat = {data.get('coord', {}).get('lat')}\n"

    if data and 'list' in data and data['list']:
        air_data = data['list'][0]
        formatted_string += f"AQI: {air_data.get('main',{}).get('aqi')}\n"
        components = air_data.get('components',{})
        formatted_string += f"Pollutants:\n"
        formatted_string += f"  PM2.5: {components.get('pm2_5')} μg/m³\n" if components.get('pm2_5') else "  PM2.5: Not Available\n"
        formatted_string += f"  PM10: {components.get('pm10')} μg/m³\n" if components.get('pm10') else "  PM10: Not Available\n"
        formatted_string += f"  CO: {components.get('co')} μg/m³\n" if components.get('co') else "  CO: Not Available\n"
        formatted_string += f"  NO: {components.get('no')} μg/m³\n" if components.get('no') else "  NO: Not Available\n"
        formatted_string += f"  NO2: {components.get('no2')} μg/m³\n" if components.get('no2') else "  NO2: Not Available\n"
        formatted_string += f"  O3: {components.get('o3')} μg/m³\n" if components.get('o3') else "  O3: Not Available\n"
        formatted_string += f"  SO2: {components.get('so2')} μg/m³\n" if components.get('so2') else "  SO2: Not Available\n"
        formatted_string += f"  NH3: {components.get('nh3')} μg/m³\n" if components.get('nh3') else "  NH3: Not Available\n"
        formatted_string += f"Timestamp: {air_data.get('dt')}\n"
    return formatted_string


# ----- Gemini Interaction -----
def get_gemini_response(query, formatted_data, city_name, detail_level):
    """
       Generates a response using Gemini, providing a conversational feel.

       Parameters:
           query (str): User's input.
           formatted_data (str): String of all values from the api
           city_name (str): The name of the city
           detail_level (str): How much information should be provided, such as "full" or "concise"

        Returns:
           str: The response from Gemini
    """
    if not formatted_data:
        return "I could not get the air quality data, try again later!"

    prompt = f"""
        You are a helpful personal assistant named AirSense AI that helps users understand air quality.
        The current location is {city_name}.
        """
    if detail_level == "concise":
        prompt += f"""
        Based on the current air quality data for {city_name}, provide a short summary with the Air Quality Index (AQI), and whether it is considered good or bad air quality, also include PM2.5 and PM10 values and say whether or not it is safe to be outside. Do not include any other values
        Here is the full air quality data:
        {formatted_data}
       """
    elif detail_level == "full":
       prompt += f"""Here is the full air quality data including coordinates, all available pollutants and a timestamp:
        {formatted_data}
        Based on this air quality data for {city_name}, answer the user query: '{query}'.
        If the user question is broad or general, provide suggestions like:
        - What activities are safe today?
        - Where is the air quality good?
        - How does the air quality affect my health?
        - What are the AQI and pollutant values?
        - What does the AQI level mean?
        Provide information about what the air quality values mean and how they might affect them.
        If the query is not relevant to air quality, please say "I can help you with air quality related questions".
       """
    prompt += f" Answer the user query: '{query}'."


    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.exception(f"Error from Gemini:")
        return "I'm sorry, I encountered an error. Please try again later."


# ----- Display Data Function -----
def display_air_quality_data(data, city_name, detail_level="concise"):
    """
       Formats and prints the Air Quality Data.

       Parameters:
           data (dict): A dictionary of all air quality data.
           city_name (str): City name
           detail_level (str): How much information should be provided, such as "full" or "concise"

       Returns:
           None
    """
    if detail_level == "concise":
      return #Do not output anything

    print(f"Air Quality Data for: {city_name}")
    if data and 'list' in data and data['list']:
      air_data = data['list'][0]
      aqi = air_data.get('main', {}).get('aqi')
      if aqi is not None:
         print(f"- AQI: {aqi} ({get_activity_recommendation(aqi).split('.')[0]})")
      components = air_data.get('components',{})
      if components:
          if components.get('pm2_5') is not None:
              print(f"- PM2.5: {components.get('pm2_5'):.2f} μg/m³")
          if components.get('pm10') is not None:
              print(f"- PM10: {components.get('pm10'):.2f} μg/m³")


      other_pollutants = []
      for key in ["co", "no", "no2", "o3", "so2", "nh3"]:
          if components.get(key) is not None:
             other_pollutants.append(f"{key.upper()}: {components.get(key):.2f} μg/m³")

      if other_pollutants:
          print("- Other Pollutants:")
          for pollutant in other_pollutants:
             print(f"  - {pollutant}")

# ----- IOT api Request function -----
def get_iot_data():
    """
      Fetches air quality data from the IOT device, calculates AQI, and returns it with status.

      Returns:
          dict or None: Air quality data with calculated AQI and status, or None if an error occurs.
    """
    try:
        response = requests.get(IOT_API_URL)
        response.raise_for_status()
        data = response.json()
        if data:
            print (f"get_iot_data(): Successfully retrieved IoT data: {data}")
            pm25 = data.get("pm2_5")
            pm10 = data.get("pm10")
            if pm25 is not None and pm10 is not None:
                aqi, status = calculate_aqi_and_status(pm25, pm10)
                print(f"get_iot_data(): Successfully calculated aqi: {aqi} and status: {status}")
                return {"pm2_5": pm25, "pm10": pm10, "aqi": aqi, "status": status}
            else:
                 logging.error(f"get_iot_data(): No pm2_5 or pm10 values in IoT data")
                 return None
        else:
           logging.error(f"get_iot_data(): No IoT data available ")
           return None
    except requests.exceptions.RequestException as e:
      logging.error(f"get_iot_data(): Error fetching IOT data: {e}")
      return None
    
def calculate_aqi_and_status(pm25, pm10):
    """
      Calculates AQI and determines the status.

        Parameters:
            pm25 (float): PM2.5 value.
            pm10 (float): PM10 value.

        Returns:
            tuple: A tuple containing the (aqi, status).
    """
    # Simplified AQI calculation for demonstration purposes
    # You might want to implement different calculations based on the location you are using.
    aqi =  (pm25 + pm10) # Simple example
    status = "Good"
    if aqi > 50 :
        status = "Moderate"
    if aqi > 100 :
        status = "Unhealthy"
    if aqi > 150 :
        status = "Harmful"
    if aqi > 200:
       status = "Hazardous"

    return aqi, status


# ---- New API Endpoint ------
@app.route('/aqipollutants', methods=['GET'])
def aqi_pollutants_api():
    """
    Returns AQI, other pollutants, PM values, and an air quality status as text values in a JSON response.
    """
    global current_latitude, current_longitude
    air_quality_data = get_air_quality_data(current_latitude, current_longitude)
    if air_quality_data and air_quality_data['list']:
        try:
            air_data = air_quality_data['list'][0]
            components = air_data.get('components', {})
            aqi = air_data.get('main', {}).get('aqi')
            pm25 = components.get('pm2_5')
            pm10 = components.get('pm10')
            co = components.get('co')
            no = components.get('no')
            no2 = components.get('no2')
            o3 = components.get('o3')
            so2 = components.get('so2')
            nh3 = components.get('nh3')

            aqi_status = "Not Available"
            if aqi is not None:
                 if aqi <= 1:
                    aqi_status = "Good"
                 elif aqi <= 2:
                   aqi_status = "Moderate"
                 elif aqi <= 3:
                    aqi_status = "Unhealthy for sensitive groups"
                 elif aqi <= 4:
                    aqi_status = "Unhealthy"
                 elif aqi <= 5:
                     aqi_status = "Very Unhealthy"
                 elif aqi > 5:
                    aqi_status = "Hazardous"


            if aqi is not None:
               return jsonify({
                   "aqi": f"{aqi}",
                    "pm25": f"{pm25}" if pm25 is not None else "Not Available",
                   "pm10": f"{pm10}" if pm10 is not None else "Not Available",
                    "co": f"{co}" if co is not None else "Not Available",
                    "no": f"{no}" if no is not None else "Not Available",
                    "no2": f"{no2}" if no2 is not None else "Not Available",
                    "o3": f"{o3}" if o3 is not None else "Not Available",
                    "so2": f"{so2}" if so2 is not None else "Not Available",
                    "nh3": f"{nh3}" if nh3 is not None else "Not Available",
                   "aqi_status": f"{aqi_status}",
                    }), 200
            else:
               return jsonify({"error": "AQI value not available"}), 400
        except Exception as e:
            logging.error(f"aqi_pollutants_api(): An error has occurred: {e}")
            return jsonify({"error": "Could not process air quality data"}), 500

    else:
        return jsonify({"error": "Could not get air quality data"}), 500

# ---- New API Endpoint ------
@app.route('/pmairquality', methods=['GET'])
def pm_air_quality_api():
    """
    Returns only PM2.5 and PM10 values as a text response.
    """
    global current_latitude, current_longitude
    air_quality_data = get_air_quality_data(current_latitude, current_longitude)
    if air_quality_data and air_quality_data['list']:
        try:
           air_data = air_quality_data['list'][0]
           components = air_data.get('components', {})
           pm25 = components.get('pm2_5')
           pm10 = components.get('pm10')
           if pm25 is not None and pm10 is not None:
              return jsonify({"pm25": f"{pm25}", "pm10": f"{pm10}"}), 200
           else:
             return jsonify({"error": "PM2.5 or PM10 values not available"}), 400
        except Exception as e:
          logging.error(f"pm_values_api(): An error has occured: {e}")
          return jsonify({"error": "Could not process air quality data"}), 500
    else:
        return jsonify({"error": "Could not get air quality data"}), 500
    
# ----- New API Endpoint to get pm values and status ------
@app.route('/aqipollutantsiot', methods=['GET'])
def aqi_pollutants_api():
    """
    Returns AQI, other pollutants, PM values, and an air quality status as text values in a JSON response.
    """
    iot_data = get_iot_data()  #Get data from IOT API

    if iot_data is not None:

       return jsonify({
                   "aqi": f"{iot_data.get('aqi')}",
                    "pm25": f"{iot_data.get('pm2_5')}" if iot_data.get('pm2_5') is not None else "Not Available",
                   "pm10": f"{iot_data.get('pm10')}" if iot_data.get('pm10') is not None else "Not Available",
                  "aqi_status": f"{iot_data.get('status')}",
                    }), 200
    else:
      return jsonify({"error": "Could not get data from IoT device"}), 500



@app.route('/', methods=['GET', 'POST'])
def air_quality_api():
    """
       Main function to return air quality data to a user using a Flask API
    """
    global current_latitude, current_longitude, current_city_name

    if request.method == 'GET':
        user_query = request.args.get('query', '').lower()
    elif request.method == 'POST':
        data = request.get_json()
        user_query = data.get('query', '').lower()
    else:
        return jsonify({"error": "Method not allowed"}), 405

    # Explicit Location Change
    location_match = re.search(r"change the location to ([\w\s]+(?:,\s*\w+)?)", user_query, re.IGNORECASE)
    if location_match:
        city_name = location_match.group(1).strip()
        print(f"main(): identified city for location change request: {city_name}")  # Debug message
        coordinates = get_coordinates(city_name)
        if coordinates:
            current_latitude, current_longitude = coordinates
            current_city_name = city_name
            print(f"main(): Updated location: current_city_name: {current_city_name}, current_latitude: {current_latitude}, current_longitude: {current_longitude}") # Debug message
            return jsonify({"status": "Location updated to " + current_city_name})
        else:
            return jsonify({"error": f"Could not determine coordinates for city {city_name}. Using previous location."}), 400

    air_quality_data = get_air_quality_data(current_latitude, current_longitude)
    if air_quality_data:
      formatted_data = format_api_response(air_quality_data)
      if "details" in user_query or "pollutants" in user_query:
        display_air_quality_data(air_quality_data, current_city_name, "full")
        response = get_gemini_response(user_query, formatted_data, current_city_name, "full")
      else:
        display_air_quality_data(air_quality_data, current_city_name, "concise")
        response = get_gemini_response(user_query, formatted_data, current_city_name, "concise")

      return jsonify({"response": response})
    else:
       return jsonify({"error": "Could not get air quality data"}), 500


# ----- Entry Point Execution -----
if __name__ == "__main__":
    app.run(debug=True)