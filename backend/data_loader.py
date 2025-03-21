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
            'index_to_anime_id.json': '1e6qHlxDtMRQDLIu2AWMqfORSeqJHtcji'
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_file(self, file_id: str, destination: str):
        """Download a file from Google Drive."""
        print(f"Downloading file with ID: {file_id}")
        
        # Direct download URL for large files
        url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0&confirm=t'
        
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        try:
            response = session.get(url, headers=headers, stream=True)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                
                # Validate JSON
                if destination.endswith('.json'):
                    with open(destination, 'r') as f:
                        json.load(f)
                    print("JSON validation successful")
            else:
                raise Exception(f"Download failed with status code: {response.status_code}")
                
        except Exception as e:
            print(f"Error downloading file: {str(e)}")
            if os.path.exists(destination):
                os.remove(destination)
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