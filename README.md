# BIP Analityk — Toruń

A full-stack application that analyzes Biuletyn Informacji Publicznej (BIP) documents,
geocodes addresses using OpenAI LLM API + NER, and shows nearby bulletins on an interactive map.

<img width="1137" height="586" alt="image" src="https://github.com/user-attachments/assets/abf78b78-f83d-411c-8be7-8dfee26cb75b" />


## Architecture

```
bip_app/
├── requirements.txt
├── README.md        
└── app 
    ├── main.py      # FastAPI backend
    └── index.html   # Single-page frontend (served by FastAPI)
```

## Setup

```bash
pip install -r requirements.txt
export LLM_API_KEY="your-key-here"
export DISTANCE=10        # Default search radius in km (optional)
```

## Run

```bash
cd bip_app
python app/main.py
# → Open http://localhost:8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/user_input` | Geocode addresses using LLM NER → `{user_addresses, coordinates}` |
| GET | `/download_bip` | Scrape BIP from last month, analyze with LLM, cache in memory |
| POST | `/get_nearest_bip` | Return BIP entries within `distance_km` of user addresses |
| GET | `/` | Serve the browser frontend |

### POST /user_input
```json
{ "user_addresses": ["ul. Szeroka 12, Toruń", "ul. Kopernika 5"] }
→ { "user_addresses": [...], "coordinates": [[53.01, 18.59], ...] }
```

### POST /get_nearest_bip
```json
{ "user_addresses": ["ul. Szeroka 12"], "distance_km": 5 }
→ { "user_coordinates": [...], "bip_entries": [{...}] }
```

### BIP Entry format
```json
{
  "id": 1,
  "typ": "uchwała",
  "numer": "BIP-0001",
  "data": "2026-02-15",
  "data_wejscia": "2026-02-15",
  "tytuł": "...",
  "coordinates": [53.0138, 18.5981],
  "impact": "medium",
  "type": ["noise", "more traffic"],
  "address": "ul. Przykładowa 1, Toruń",
  "url": "https://...",
  "distance_km": 1.234
}
```

## Notes

- BIP scraping uses OPenAI API to extract addresses and assess impact from bulletin text
- Geocoding uses Open Street Map for geocoding
- The frontend communicates with the backend on the same origin (port 8000)
- Distances computed with the Haversine formula
- Set `DISTANCE` env var to change the default search radius
