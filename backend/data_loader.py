import os
import json
import requests
from pathlib import Path

class DataLoader:
    """Handles downloading and caching of data files from Google Drive."""
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.file_ids = {
            'final_anime_data.json': '1yHRbt5_bp_HziYvAKYKaoQ3vBBk4-roe',
            'anime_to_reviewer.json': '1scpa4SO_hXQGiU9RZH0a2Nt4OJMSdmIe',
            'reviewer_to_anime.json': '1rNpIj4Q4ZXECBKd9N1Hk2IiWMrTZ6Piu',
            'anime_to_index.json': '1LLStd02z3riZCt9f7SJyET3L5OJxmjKL',
            'index_to_anime_id.json': '1e6qHlxDtMRQDLIu2AWMqfORSeqJHtcj'
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_file(self, file_id: str, destination: str):
        """Download a file from Google Drive with proper handling of confirmation tokens."""
        print(f"Attempting to download file with ID: {file_id}")
        
        # For large files, we need to use a different approach
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        # Start a session to handle cookies
        session = requests.Session()
        
        # Make initial request
        response = session.get(url, stream=True)
        print(f"Initial response status: {response.status_code}")
        
        # Check if we're getting the confirmation page
        if "confirm=t" in response.text or "confirm=" in response.text or 'Virus scan warning' in response.text:
            print("Detected confirmation page. Extracting confirmation token...")
            
            # Try to find the confirmation token
            confirmation_token = None
            for line in response.text.splitlines():
                if 'confirm=' in line:
                    start_idx = line.find('confirm=') + len('confirm=')
                    end_idx = line.find('&', start_idx) if '&' in line[start_idx:] else len(line)
                    confirmation_token = line[start_idx:end_idx]
                    print(f"Found token: {confirmation_token}")
                    break
            
            if confirmation_token:
                # Use the confirmation token to get the actual file
                url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={confirmation_token}"
                response = session.get(url, stream=True)
                print(f"Confirmed download response status: {response.status_code}")
            else:
                # Alternative method using the cookies
                print("Using cookie-based confirmation...")
                params = {'id': file_id, 'confirm': 't'}
                response = session.get("https://drive.google.com/uc?export=download", params=params, stream=True)
                print(f"Cookie confirmation response status: {response.status_code}")
        
        # Save the file
        CHUNK_SIZE = 32768
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        
        print(f"File saved to: {destination}")
        
        # Validate JSON
        if destination.endswith('.json'):
            try:
                with open(destination, 'r', encoding='utf-8') as f:
                    json_content = f.read()
                    # Check if the content appears to be HTML
                    if '<html' in json_content.lower() or '<!doctype html' in json_content.lower():
                        print("Error: Received HTML content instead of JSON. The download might have failed.")
                        raise ValueError("Downloaded content is HTML, not valid JSON")
                    
                    # Try to parse the JSON
                    json.loads(json_content)
                print("JSON validation successful")
            except Exception as e:
                print(f"Error validating JSON: {str(e)}")
                raise
    
    def load_json_file(self, filename: str):
        """Load a JSON file directly from Google Drive."""
        file_path = os.path.join(self.data_dir, filename)
        
        print(f"Downloading {filename} from Google Drive...")
        self.download_file(self.file_ids[filename], file_path)
        
        # Load and return the JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_all_data(self):
        """Load all required data files."""
        data = {}
        for filename in self.file_ids.keys():
            print(f"Loading {filename}...")
            data[filename] = self.load_json_file(filename)
        return data