"""
BIP (Biuletyn Informacji Publicznej) Analysis API
FastAPI backend with LLM-powered address geocoding, impact analysis, and BIP scraping.
"""
import logging
import os
import json
import re
import math
import httpx
from datetime import datetime, timedelta
from typing import Literal, Optional, Tuple
from bs4 import BeautifulSoup

import asyncio
import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

# ── Configuration ──────────────────────────────────────────────────────────────
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_URL = os.getenv("LLM_API_URL", "")
DISTANCE_KM = float(os.getenv("DISTANCE", "10"))
USER_EMAIL = os.getenv("USER_EMAIL", "")
BIP_URL = "https://prawomiejscowe.pl/UrzadMiastaTorunia/tabBrowser/mainPage"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

app = FastAPI(title="BIP Analyzer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_URL)

# In-memory store for last BIP download
_last_bip: list[dict] = []

# ── Pydantic models ─────────────────────────────────────────────────────────────
class UserInputRequest(BaseModel):
    user_addresses: list[str]

class UserInputResponse(BaseModel):
    user_addresses: list[str]
    coordinates: list[tuple[float | None, float | None]]

class ImpactResponse(BaseModel):
    impact: Literal["small", "medium", "high"]
    type: list[str]

class NearestBipRequest(BaseModel):
    user_addresses: list[str]
    distance_km: float = DISTANCE_KM

# ── LLM helpers ─────────────────────────────────────────────────────────────────
# def llm_geocode_addresses(addresses: list[str]) -> list[tuple[float, float]]:
#     """Use Claude to extract / approximate coordinates for given addresses using NER."""
#     prompt = f"""You are a geocoding assistant with NER capabilities.
# Given the following list of addresses (likely in Toruń, Poland or nearby), 
# extract the named entities and return approximate GPS coordinates (latitude, longitude) for each.

# Addresses:
# {json.dumps(addresses, ensure_ascii=False)}

# Respond ONLY with a valid JSON array of [lat, lon] pairs in the same order as the input.
# Use real-world coordinates for Toruń, Poland region.
# Example format: [[53.0138, 18.5981], [53.0200, 18.6100]]
# """
#     message = client.chat.completions.create(
#         model="gemini-2.5-pro",
#         max_tokens=1024,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     text = message.content[0].text.strip()
#     # Extract JSON array
#     match = re.search(r'\[\s*\[.*?\]\s*\]', text, re.DOTALL)
#     if match:
#         coords = json.loads(match.group())
#         return [tuple(c) for c in coords]
#     raise ValueError(f"Could not parse coordinates from LLM response: {text}")

def extract_address(text: str) -> Optional[list[str]]:
    """Use LLM API to extract the geographical area name from BIP content."""
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY not set")
    
    prompt = f"""You are a named entity recognition system. 
Given the following text from a Polish public information bulletin (BIP), extract the most specific geographical area or location name that the announcement pertains to.
Return ONLY the location name (e.g. "Warszawa, ul. Marszałkowska", "gmina Piaseczno", "powiat wołomiński") - nothing else.
If multiple locations are mentioned, return the json with list of addresses.
If no clear location is found, return "UNKNOWN".

Text:
{text}"""

    try:
        message = client.chat.completions.create(
            model="gemini-3-flash-preview",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            response_format="json_schema",
        ) 
        area = message.content[0].text.strip()
        return [] if area == "UNKNOWN" else json.dumps(area)
    except Exception as e:
        print(f"[LLMService] Error: {e}")
        return None

def remove_address_prefixes(address):
    """Remove prefixes that are redundant for geocoding."""
    address_lower = address.lower()
    PREFIXES = ["ul. ", "ulica ", "os. ", "osiedle "]
    for prefix in PREFIXES:
        if prefix in address:
            print(f"found {prefix} in address {address}")
            ind = address_lower.find(prefix)
            address = address[:ind] + address[ind+len(prefix):]
            print(f"new address {address}")
    if "toruń" not in address_lower:
        address += " Toruń"
    return address

async def llm_geocode_addresses(addresses: list[str]) -> list[tuple[float, float]]:
    addresses = [remove_address_prefixes(i) for i in addresses]
    coords = await asyncio.gather(*[geocode(address) for address in addresses])
    return coords

async def geocode(address: str) -> Optional[Tuple[float | None, float | None]]:
    """Geocode an address/area name using Nominatim (OpenStreetMap)."""
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
        "countrycodes": "pl",  # bias to Poland; remove if needed
    }
    headers = {"User-Agent": f"BIPMonitor/1.0 (research project; {USER_EMAIL})"}
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(NOMINATIM_URL, params=params, headers=headers)
            print(resp)
            resp.raise_for_status()
            results = resp.json()
        
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            return lat, lon
        
        # Retry without country bias
        params.pop("countrycodes", None)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(NOMINATIM_URL, params=params, headers=headers)
            print(resp)
            resp.raise_for_status()
            results = resp.json()
        
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
        
        return (None, None)
    except Exception as e:
        print(f"[GeoService] Geocoding error for '{address}': {e}")
        return (None, None)


def llm_estimate_impact(text: str) -> ImpactResponse:
    """Use Claude to estimate impact type and magnitude from BIP entry text."""
    prompt = f"""You are an urban planning expert analyzing a public information bulletin entry.
Based on the following text, estimate:
1. The overall impact magnitude: "small", "medium", or "high"
2. The types of impact for neighboring areas (choose from: change of value, noise, smell, more traffic, less traffic, new business opportunities, environmental, construction disruption, visual change, safety)

Text:
{text[:3000]}

Respond ONLY with valid JSON in this exact format:
{{"impact": "medium", "type": ["noise", "more traffic"]}}
"""
    logging.info("Attempting to estimate impact.")
    message = client.chat.completions.create(
        model="gemini-2.5-pro",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        logging.info("Found impact %s" %match.group())
        data = json.loads(match.group())
        return ImpactResponse(**data)
    raise ValueError(f"Could not parse impact from LLM response: {raw}")


async def llm_geocode_single(address: str) -> tuple[float, float]:
    """Geocode a single address string."""
    logging.info("Attempting to geocode address: %s" %address)
    coords = await llm_geocode_addresses([address])
    logging.info("Found coordinates %s for address: %s" %(coords, address))
    return coords[0]

# ── BIP scraper ─────────────────────────────────────────────────────────────────
async def scrape_bip_entries() -> list[dict]:
    """Download and parse BIP entries from the last month."""
    one_month_ago = datetime.now() - timedelta(days=31)
    entries = []
    logging.info("Attempting to scrape BIP entries.")

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as http:
        try:
            resp = await http.get(BIP_URL)
            print(resp)
            resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch BIP page: {e}")

        soup = BeautifulSoup(resp.text, "html.parser")

        # Try to find table rows with BIP data
        logging.info("Attempting to use soup.")
        rows = soup.find_all("tr")
        print("rows", rows)
        
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 4:
                continue
            
            cell_texts = [c.get_text(strip=True) for c in cells]
            
            # Skip header rows
            if any(h in cell_texts[0].lower() for h in ["id", "lp", "#", "numer"]):
                continue

            # Try to parse a date from any cell
            entry_date = None
            date_str = ""
            for ct in cell_texts:
                for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y"):
                    try:
                        entry_date = datetime.strptime(ct[:10], fmt)
                        date_str = ct[:10]
                        break
                    except ValueError:
                        continue
                if entry_date:
                    break

            if entry_date and entry_date < one_month_ago:
                continue  # skip old entries

            # Extract a link if present
            links = row.find_all("a", href=True)
            detail_text = " ".join(cell_texts)
            detail_url = None
            
            if links:
                href = links[0]["href"]
                if href.startswith("http"):
                    detail_url = href
                else:
                    from urllib.parse import urljoin
                    detail_url = urljoin(BIP_URL, href)

            # Fetch detail page for more text
            full_text = detail_text
            if detail_url:
                try:
                    det_resp = await http.get(detail_url)
                    det_soup = BeautifulSoup(det_resp.text, "html.parser")
                    full_text = det_soup.get_text(separator=" ", strip=True)[:5000]
                except Exception:
                    pass

            # Try to find an address in the text using Claude
            address_prompt = f"""Extract the street address(es) in Toruń, Poland from this text.
Text: {full_text[:1000]}
Return ONLY a JSON array of address strings, e.g. ["ul. Szeroka 12, Toruń"].
If no address found, return []."""
            
            try:
                print("sending request to LLM to ask for place NER")
                addr_msg = client.messages.create(
                    model="gemini-3-flash-preview",
                    max_tokens=256,
                    messages=[{"role": "user", "content": address_prompt}],
                )
                addr_raw = addr_msg.content[0].text.strip()
                addr_match = re.search(r'\[.*?\]', addr_raw, re.DOTALL)
                found_addresses = json.loads(addr_match.group()) if addr_match else []
            except Exception:
                found_addresses = []

            if not found_addresses:
                found_addresses = ["Toruń, Poland"]  # fallback

            # Get impact
            try:
                impact_data = llm_estimate_impact(full_text)
            except Exception:
                impact_data = ImpactResponse(impact="small", type=["unknown"])

            # Build one entry per address
            for addr in found_addresses:
                try:
                    coords = llm_geocode_single(addr)
                except Exception:
                    coords = (53.0138, 18.5981)  # Toruń center fallback

                entry = {
                    "id": len(entries) + 1,
                    "typ": cell_texts[1] if len(cell_texts) > 1 else "",
                    "numer": cell_texts[0] if cell_texts else "",
                    "data": date_str,
                    "data_wejscia": date_str,
                    "tytuł": cell_texts[2] if len(cell_texts) > 2 else detail_text[:100],
                    "coordinates": coords,
                    "impact": impact_data.impact,
                    "type": impact_data.type,
                    "address": addr,
                    "url": detail_url or BIP_URL,
                }
                entries.append(entry)

        # If we got nothing from table parsing, create sample entries
        # (the website structure may vary; this ensures the API always returns something useful)
        if not entries:
            # Try alternative parsing - look for any structured list items
            items = soup.find_all(["li", "article", "div"], class_=re.compile(r'item|entry|row|record', re.I))
            for i, item in enumerate(items[:20]):
                text_content = item.get_text(separator=" ", strip=True)
                if len(text_content) < 20:
                    continue
                try:
                    impact_data = llm_estimate_impact(text_content)
                except Exception:
                    impact_data = ImpactResponse(impact="small", type=["general"])
                
                entries.append({
                    "id": i + 1,
                    "typ": "document",
                    "numer": f"BIP-{i+1:04d}",
                    "data": datetime.now().strftime("%Y-%m-%d"),
                    "data_wejscia": datetime.now().strftime("%Y-%m-%d"),
                    "tytuł": text_content[:120],
                    "coordinates": (53.0138 + (i * 0.002), 18.5981 + (i * 0.002)),
                    "impact": impact_data.impact,
                    "type": impact_data.type,
                    "address": "Toruń, Poland",
                    "url": BIP_URL,
                })

    return entries

# ── Distance helper ──────────────────────────────────────────────────────────────
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in km between two GPS points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── Endpoints ───────────────────────────────────────────────────────────────────
@app.post("/user_input", response_model=UserInputResponse)
async def user_input(body: UserInputRequest):
    """Transform address list to addresses + coordinates using LLM NER."""
    if not body.user_addresses:
        raise HTTPException(status_code=400, detail="No addresses provided")
    try:
        coords = await llm_geocode_addresses(body.user_addresses)
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
    print("User input: %s" %body.user_addresses)
    print("Coordinates: %s" %coords)
    return UserInputResponse(user_addresses=body.user_addresses, coordinates=coords)


@app.get("/download_bip")
async def download_bip():
    """Download and organize BIP information from the last month."""
    global _last_bip
    entries = await scrape_bip_entries()
    _last_bip = entries
    return entries


@app.post("/get_nearest_bip")
async def get_nearest_bip(body: NearestBipRequest):
    """Return BIP entries within distance_km of any user address."""
    print(_last_bip)
    if not _last_bip:
        raise HTTPException(status_code=404, detail="No BIP data loaded. Call /download_bip first.")
    
    try:
        user_coords = await llm_geocode_addresses(body.user_addresses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    for entry in _last_bip:
        bip_lat, bip_lon = entry["coordinates"]
        min_dist = min(
            haversine(uc[0], uc[1], bip_lat, bip_lon)
            for uc in user_coords
        )
        if min_dist <= body.distance_km:
            results.append({**entry, "distance_km": round(min_dist, 3)})

    results.sort(key=lambda x: x["distance_km"])
    return {"user_coordinates": user_coords, "bip_entries": results}


@app.get("/")
def serve_frontend():
    return FileResponse("./index.html")


# ── Run ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")
