import requests
import os
import time

# ================= CONFIGURATION =================
# API keys must not be present in any open source platforms. That's why I have removed that.
API_KEY = ""  # <--- PASTE KEY HERE
BASE_DIR = "dataset_raw"
LIMIT_PER_TERM = 50 

# THE MASTER EMOTION LIST

SEARCH_TERMS = {
    "happy":    ["laugh", "giggle", "yay", "cheer", "clapping"],
    "angry":    ["yell", "argument", "rage", "grunting", "slam"],
    "fear":     ["scream", "gasp", "panic", "terror", "scared_breathing"],
    "pain":     ["groan", "ouch", "pain_cry", "hurt"],
    "sad":      ["crying", "sob", "weeping", "whimper", "sniffle"],
    "disgust":  ["retching", "vomit", "eww", "blech"],
    "boredom":  ["yawn", "sigh", "monotone", "bored_voice"],
    "surprise": ["gasp", "wow", "omg", "shock"],
    "confused": ["huh", "hmm", "what"],
    "neutral":  ["counting", "reading_book", "speaking_calm", "monologue"]
}


def download_file(url, folder, filename):
    try:
        # Get the file content directly
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(folder, filename), 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"      ! Download Error: {e}")
    return False

def run_collection():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    print(f"--- Starting DIRECT API Collection (No Library) ---")
    
    headers = {'Authorization': f'Token {API_KEY}'}

    for sentiment, queries in SEARCH_TERMS.items():
        print(f"\n>>> PROCESSING: {sentiment.upper()}")
        category_dir = os.path.join(BASE_DIR, sentiment)
        os.makedirs(category_dir, exist_ok=True)
        
        for query in queries:
            print(f"  > Searching: '{query}'")
            
            # 1. Build the Search URL Manually
            search_url = "https://freesound.org/apiv2/search/text/"
            params = {
                'query': query,
                'filter': "duration:[0.5 TO 6.0]",
                'fields': "id,name,previews",
                'page_size': LIMIT_PER_TERM + 10,
                'token': API_KEY # Sending token in params just in case
            }
            
            try:
                # 2. Call API
                resp = requests.get(search_url, params=params, headers=headers)
                
                if resp.status_code != 200:
                    print(f"    ! API Error {resp.status_code}: {resp.text}")
                    continue
                
                data = resp.json()
                results = data.get('results', [])
                
                # 3. Process & Download
                count = 0
                for sound in results:
                    if count >= LIMIT_PER_TERM: break
                    
                    # Extract the High-Quality MP3 link
                    if 'previews' in sound and 'preview-hq-mp3' in sound['previews']:
                        mp3_url = sound['previews']['preview-hq-mp3']
                        
                        safe_name = "".join([c if c.isalnum() else "_" for c in sound['name']])
                        filename = f"{query}_{sound['id']}_{safe_name}.mp3"
                        filepath = os.path.join(category_dir, filename)
                        
                        if not os.path.exists(filepath):
                            success = download_file(mp3_url, category_dir, filename)
                            if success:
                                count += 1
                                time.sleep(0.5)
                        else:
                            # File exists
                            pass
                
                print(f"Saved {count} files")
                
            except Exception as e:
                print(f"Critical Error: {e}")

if __name__ == "__main__":
    run_collection()