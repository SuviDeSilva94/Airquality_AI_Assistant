import openai
from openai import OpenAI
import requests

# Initialize OpenAI client
try:
    client = OpenAI(api_key='')
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    raise

# Function to get air quality data
def get_air_quality():
    try:
        # Replace with your OpenWeather API key
        api_key = ''
        lat = 35.6895  # Example: Latitude for Tokyo
        lon = 139.6917  # Example: Longitude for Tokyo

        print("Fetching air quality data...")
        # Make API request
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        print(f"Requesting URL: {url}")
        response = requests.get(url)

        # Raise error for non-200 status codes
        response.raise_for_status()

        # Extract AQI from the response
        data = response.json()
        print("Air quality data fetched successfully.")
        return data['list'][0]['main']['aqi']
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
        return None
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"General Request error occurred: {req_err}")
        return None
    except KeyError as key_err:
        print(f"Key error while parsing air quality data: {key_err}")
        return None

# Function to get GPT response based on query and air quality
def get_gpt_response_with_air_quality(user_input):
    air_quality = get_air_quality()
    if air_quality is None:
        return "Sorry, I couldn't retrieve the air quality data. Please try again later."

    print(f"Current air quality (AQI): {air_quality}")
    # Construct the prompt with air quality information
    prompt = f"User asks: {user_input}\nCurrent air quality (AQI): {air_quality}\nRespond based on the air quality levels."

    try:
        print("Fetching response from OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can use "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are an AI assistant providing air quality advice."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        print("Response received from OpenAI API.")
        return response.choices[0].message.content.strip()
    except openai.error.AuthenticationError as auth_err:
        print(f"Authentication error with OpenAI API: {auth_err}")
        return "Sorry, there was an authentication error. Please check the API key."
    except openai.error.RateLimitError as rate_err:
        print(f"Rate limit error with OpenAI API: {rate_err}")
        return "Sorry, you've exceeded the rate limit. Please try again later."
    except openai.error.OpenAIError as general_err:
        print(f"General OpenAI API error: {general_err}")
        return "Sorry, I couldn't process your request. Please try again later."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An unexpected error occurred. Please try again."

# Example usage
if __name__ == "__main__":
    user_query = "Can I go jogging today?"
    print(f"User query: {user_query}")
    response = get_gpt_response_with_air_quality(user_query)
    print(f"AI Response: {response}")
