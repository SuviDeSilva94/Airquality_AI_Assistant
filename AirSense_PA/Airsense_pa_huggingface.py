from transformers import pipeline
import requests
import time

# Function to get air quality data
def get_air_quality(api_key, lat=35.6895, lon=139.6917):
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        response = requests.get(url)

        # Raise error for non-200 status codes
        response.raise_for_status()

        # Extract AQI from the response
        data = response.json()
        return data['list'][0]['main']['aqi']  # Air Quality Index (AQI)
    
    except requests.exceptions.RequestException as err:
        print(f"Error fetching air quality data: {err}")
        return None


# Function to generate GPT-2 response based on query and air quality data
def get_gpt_response_with_air_quality(user_input, air_quality, generator):
    if air_quality is None:
        return "Sorry, I couldn't retrieve the air quality data. Please try again later."

    print(f"Current air quality (AQI): {air_quality}")
    # Construct the prompt with air quality information
    prompt = f"User asks: {user_input}\nCurrent air quality (AQI): {air_quality}\nRespond based on the air quality levels."

    # Generate text using Hugging Face model pipeline
    text = generator(prompt, max_length=150, num_return_sequences=1, pad_token_id=50256)

    return text


# Main execution
if __name__ == "__main__":
    # Start the time tracking
    start = time.time()
    print("Time elapsed on working...")

    # Initialize the Hugging Face pipeline with the GPT-Neo model
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

    # OpenWeather API Key
    api_key = ''  # Replace with your OpenWeather API key

    # Example user query
    user_query = "Can I go jogging today?"

    # Get air quality data
    air_quality = get_air_quality(api_key)

    # Generate the response with air quality consideration
    response = get_gpt_response_with_air_quality(user_query, air_quality, generator)

    # Print the generated response
    print(f"AI Response: {response}")

    # Sleep for a brief moment to simulate real-time processing
    time.sleep(0.9)

    # End time tracking and print the elapsed time
    end = time.time()
    print("Time consumed in working: ", end - start)
