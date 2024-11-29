import requests

# Hugging Face API Endpoint and Token
API_URL = "https://api-inference.huggingface.co/models/shashikumar1998/Llama-3.2-3B-Instruct"
API_TOKEN = "YOUR_HUGGINGFACE_API_KEY"  # Replace with your Hugging Face API token

# Set headers with the Authorization token
headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

def query_huggingface_api(inputs: str, parameters: dict = None) -> str:
    """
    Query the Hugging Face Inference API with input text and parameters.

    Args:
        inputs (str): Input text or query for the model.
        parameters (dict): Optional parameters for text generation.
            Example:
                {
                    "max_length": 200,
                    "temperature": 0.7,
                    "top_p": 0.95
                }

    Returns:
        str: Model's generated response or error message.
    """
    # Prepare payload for the API
    payload = {
        "inputs": inputs,
        "parameters": parameters or {}
    }

    try:
        # Send POST request to Hugging Face API
        response = requests.post(API_URL, headers=headers, json=payload)

        # Check for successful response
        if response.status_code == 200:
            # Parse and return the generated text
            response_data = response.json()
            if isinstance(response_data, list) and "generated_text" in response_data[0]:
                return response_data[0]["generated_text"]
            else:
                return "Unexpected response format from the API."
        else:
            # Print error details for debugging
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return f"Error: Unable to generate response (status code {response.status_code})."
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to connect to the API. {str(e)}"

# Main function to demonstrate usage
def main():
    """
    Main function to demonstrate querying the Hugging Face Inference API.
    """
    # Example input query
    input_query = "What is the best investment strategy? Answer like Robert."

    # Text generation parameters
    parameters = {
        "max_length": 200,  # Maximum tokens in the generated response
        "temperature": 0.7,  # Adjusts randomness in generation
        "top_p": 0.95  # Controls nucleus sampling
    }

    # Query the API
    print("Sending request to Hugging Face API...")
    response = query_huggingface_api(input_query, parameters)

    # Print the generated response
    print("\nGenerated Response:")
    print(response)

# Entry point for the script
if __name__ == "__main__":
    main()
