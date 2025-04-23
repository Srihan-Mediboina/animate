import json
from collections import defaultdict

def analyze_studios():
    # Load the anime data
    with open('backend/data/final_anime_data.json', 'r') as f:
        anime_data = json.load(f)
    
    # Create a dictionary to count anime per studio
    studio_counts = defaultdict(int)
    
    # Count anime for each studio
    for anime in anime_data:
        studios = anime.get('Studios', '')
        if studios:
            # Split by comma and strip whitespace
            studio_list = [studio.strip() for studio in studios.split(',')]
            for studio in studio_list:
                studio_counts[studio] += 1
    
    # Sort studios by count in descending order
    sorted_studios = sorted(studio_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print results
    print("\nStudio Analysis:")
    print("-" * 50)
    print(f"{'Studio':<40} {'Anime Count':>10}")
    print("-" * 50)
    for studio, count in sorted_studios:
        print(f"{studio:<40} {count:>10}")
    print("-" * 50)
    print(f"Total unique studios: {len(studio_counts)}")

if __name__ == "__main__":
    analyze_studios() 