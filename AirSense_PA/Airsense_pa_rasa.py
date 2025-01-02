import requests
from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize Hugging Face model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token to eos_token if not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Check if a GPU (MPS for macOS) is available
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class ActionGetAirQualityAndRespond(Action):
    def name(self):
        return "action_get_air_quality_and_respond"

    def run(self, dispatcher, tracker, domain):
        # Get the user query
        user_input = tracker.latest_message.get('text')

        # Get air quality data
        air_quality = self.get_air_quality()

        if air_quality is None:
            dispatcher.utter_message("Sorry, I couldn't retrieve the air quality data. Please try again later.")
            return []

        # Construct the prompt for the GPT model
        prompt = f"User asks: {user_input}\nCurrent air quality (AQI): {air_quality}\nRespond based on the air quality levels."

        # Generate the response using Hugging Face model
        response_text = self.get_gpt_response(prompt)

        # Send the generated response back to the user
        dispatcher.utter_message(response_text)

        return []

    def get_air_quality(self):
        try:
            # Replace with your OpenWeather API key
            api_key = ''
            lat = 35.6895  # Latitude for Tokyo
            lon = 139.6917  # Longitude for Tokyo

            # Make API request to get air quality data
            url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            return data['list'][0]['main']['aqi']

        except requests.exceptions.RequestException:
            return None

    def get_gpt_response(self, prompt):
        try:
            # Tokenize the input and ensure padding and attention masks are set
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Generate output from the model
            outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)

            # Decode the generated text
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response_text

        except Exception as e:
            return "Sorry, I couldn't process your request. Please try again later."
