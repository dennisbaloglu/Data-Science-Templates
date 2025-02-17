import requests

def fetch_todo():
    # URL for a sample to-do item from JSONPlaceholder API
    url = "https://jsonplaceholder.typicode.com/todos/1"
    
    try:
        # Send a GET request to the API
        response = requests.get(url)
        
        # Raise an HTTPError if the response status is 4xx or 5xx
        response.raise_for_status()
        
        # Parse the JSON response into a Python dictionary
        todo_item = response.json()
        
        # Print the retrieved data
        print("Fetched To-Do Item:")
        print(todo_item)
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")

if __name__ == "__main__":
    fetch_todo()
