import os
import urllib.request

test_dir = os.path.join(os.path.dirname(__file__), "test_images")
os.makedirs(test_dir, exist_ok=True)

# Direct sample URLs for realistic tests
urls = {
    # Fall detection sample (person lying down on floor)
    "fall_sample.jpg": "https://images.unsplash.com/photo-1541829070764-84a7d30dd3f3?w=800&q=80",
    # Sitting person (sitting on chair upright)
    "sitting_sample.jpg": "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=800&q=80",
    # Fight / physical confrontation (two people in aggressive posture)
    "fight_sample.jpg": "https://images.unsplash.com/photo-1517838277536-f5f99be501cd?w=800&q=80",
    # Indian rupee currency note (₹500 / Indian banknote)
    "cash_sample.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/500_rupee_note_specimen.jpg/640px-500_rupee_note_specimen.jpg"
}

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

for name, url in urls.items():
    dest = os.path.join(test_dir, name)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response, open(dest, "wb") as out_file:
            out_file.write(response.read())
        print(f"[OK] Downloaded {name}: {os.path.getsize(dest)} bytes")
    except Exception as e:
        print(f"[ERR] Failed {name}: {e}")
