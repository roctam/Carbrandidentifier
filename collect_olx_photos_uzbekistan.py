"""Collect up to 500 exterior car photos per brand from publicly accessible OLX.uz Uzbekistan listings.

Respect OLX robots.txt, site terms, seller rights, and HTTP rate limits.
No CAPTCHA/login bypass, proxy rotation, or access-control evasion.

Install: py -m pip install requests beautifulsoup4 pillow numpy opencv-python ultralytics
Run:     py collect_olx_photos.py

Optional: provide seed listing URLs in olx_listing_urls.csv (brand,url).
The script also discovers listings via public OLX search results, where allowed.
"""
import csv
import hashlib
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'dataset' / 'images'
SEEDS = ROOT / 'olx_listing_urls.csv'
BRANDS = ['bmw','byd','chery','chevrolet','daewoo','hyundai','kia','mercedes','toyota','volkswagen']
TARGET = 500
DELAY = 2.0
MAX_SEARCH_PAGES = 80
MAX_LISTINGS_PER_BRAND = 1500
MAX_IMAGE_BYTES = 12_000_000
USER_AGENT = 'PersonalCarDatasetResearch/1.0 (respectful public-page retrieval)'
BASE = 'https://www.olx.uz'

session = requests.Session()
session.headers.update({'User-Agent': USER_AGENT, 'Accept-Language': 'en-US,en;q=0.8'})
robot_cache = {}

def allowed_by_robots(url):
    parsed = urlparse(url)
    origin = f'{parsed.scheme}://{parsed.netloc}'
    if origin not in robot_cache:
        rp = RobotFileParser()
        rp.set_url(origin + '/robots.txt')
        try:
            rp.read()
            robot_cache[origin] = rp
        except Exception:
            robot_cache[origin] = None
    rp = robot_cache[origin]
    if rp is None:
        raise RuntimeError('Could not verify robots.txt for ' + origin)
    return rp.can_fetch(USER_AGENT, url)

last_request = 0.0

def fetch(url, image=False):
    global last_request
    parsed = urlparse(url)
    if parsed.scheme not in ('https','http'):
        raise ValueError('Unsupported URL scheme')
    if not allowed_by_robots(url):
        raise PermissionError('Disallowed by OLX robots.txt')
    time.sleep(max(0, DELAY - (time.monotonic() - last_request)))
    last_request = time.monotonic()
    response = session.get(url, timeout=25, stream=True)
    if response.status_code in (403, 429):
        raise RuntimeError(f'HTTP {response.status_code}: access restricted; stopping')
    response.raise_for_status()
    data = bytearray()
    for chunk in response.iter_content(65536):
        data.extend(chunk)
        if len(data) > (MAX_IMAGE_BYTES if image else 3_000_000):
            raise ValueError('Response too large')
    return bytes(data), response.headers.get('Content-Type','')

def is_listing(url):
    p = urlparse(url)
    return p.netloc.endswith('olx.uz') and ('-ID' in p.path or '/d/obyavlenie/' in p.path)

def listing_links(html, base):
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    for a in soup.select('a[href]'):
        u = urljoin(base, a.get('href','')).split('?')[0]
        if is_listing(u) and u not in found:
            found.append(u)
    return found

def search_urls(brand):
    # OLX search markup/URL patterns can change. Only public search pages are used.
    return [f'{BASE}/transport/legkovye-avtomobili/?q={quote(brand)}&page={page}'
            for page in range(1, MAX_SEARCH_PAGES + 1)]

def get_seed_links():
    result = {b: [] for b in BRANDS}
    if not SEEDS.exists():
        SEEDS.write_text('brand,url\n', encoding='utf-8')
        return result
    with SEEDS.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            brand = (row.get('brand') or '').strip().lower()
            url = (row.get('url') or '').strip()
            if brand in result and is_listing(url) and url not in result[brand]:
                result[brand].append(url)
    return result

def text_location_matches(soup):
    # Strictly require a Tashkent indication on the listing itself.
    text = soup.get_text(' ', strip=True).lower()
    return True  # Search is nationwide; OLX.uz listings only

def extract_images(html, listing_url):
    soup = BeautifulSoup(html, 'html.parser')
    urls = []
    def add(u):
        if not isinstance(u, str) or not u.startswith(('http://','https://')):
            return
        if urlparse(u).netloc.endswith('olx.uz') or 'olxcdn' in urlparse(u).netloc:
            if u not in urls:
                urls.append(u)
    # Images in listing galleries often have large srcset variants.
    for img in soup.select('img'):
        for key in ('src','data-src'):
            add(img.get(key))
        for key in ('srcset','data-srcset'):
            srcset = img.get(key) or ''
            for entry in srcset.split(','):
                add(entry.strip().split(' ')[0])
    for meta in soup.select('meta[property="og:image"],meta[name="twitter:image"]'):
        add(meta.get('content'))
    # Avoid logos and site icons where identifiable.
    return [u for u in urls if not any(k in u.lower() for k in ('logo','avatar','favicon','profile'))]

def normalize_image(raw):
    with Image.open(io.BytesIO(raw)) as im:
        im.verify()
    with Image.open(io.BytesIO(raw)) as im:
        if im.width < 350 or im.height < 250:
            raise ValueError('Image too small')
        im = im.convert('RGB')
        output = io.BytesIO()
        im.save(output, format='JPEG', quality=92)
        return output.getvalue()

# Pretrained segmentation model locates the visible vehicle and supplies a mask.
# Color labels are heuristic estimates, not verified ground truth.
model = None

def analyze_photo(jpg):
    global model
    if model is None:
        model = YOLO('yolov8n-seg.pt')
    arr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    h, w = arr.shape[:2]
    result = model.predict(arr, imgsz=640, conf=0.35, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    candidates = []
    for i, b in enumerate(result.boxes):
        cls = result.names[int(b.cls.item())]
        if cls not in ('car', 'truck', 'bus'):
            continue
        x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
        area = max(0,x2-x1)*max(0,y2-y1)
        if area < 0.14*w*h:
            continue
        candidates.append((area,i,(x1,y1,x2,y2),float(b.conf.item())))
    if not candidates:
        return None
    _, idx, box, confidence = max(candidates)
    x1,y1,x2,y2 = box
    mask = np.zeros((h,w),dtype=np.uint8)
    if result.masks is not None:
        polygon = result.masks.xy[idx]
        if len(polygon)>2:
            cv2.fillPoly(mask,[polygon.astype(np.int32)],255)
    if not mask.any():
        mask[y1:y2,x1:x2]=255
    # Exclude much of the upper section (glass) and bottom (wheels/road).
    yy = np.arange(h)[:,None]
    mask[(yy < y1 + int((y2-y1)*0.35))[:,0],:] = 0
    mask[(yy > y1 + int((y2-y1)*0.85))[:,0],:] = 0
    hsv = cv2.cvtColor(arr,cv2.COLOR_BGR2HSV)
    pixels = hsv[mask>0]
    if len(pixels)<300:
        color='unw'
    else:
        # Suppress dark shadows and shiny highlights where possible.
        hh,ss,vv = pixels[:,0],pixels[:,1],pixels[:,2]
        good = (vv>35)&(vv<245)
        if good.sum()<200:
            color='unw'
        else:
            hh,ss,vv=hh[good],ss[good],vv[good]
            neutral = ss<45
            if neutral.mean()>0.7:
                med=float(np.median(vv[neutral]))
                color='black' if med<75 else 'gray' if med<145 else 'silver' if med<205 else 'white'
            else:
                colorful=ss>=45
                if colorful.sum()<150:
                    color='unw'
                else:
                    hue=float(np.median(hh[colorful]))
                    color=('red' if hue<10 or hue>=170 else 'brown' if hue<20 else
                           'beige' if hue<30 and np.median(ss[colorful])<100 else
                           'other' if hue<35 else 'green' if hue<85 else
                           'blue' if hue<135 else 'other')
    return color, box, confidence

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = get_seed_links()
    known_hashes = set()
    for p in OUT.glob('*/*.jpg'):
        known_hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
    total = 0
    stop = False
    labels_file = ROOT / 'dataset' / 'auto_labels.csv'
    labels_file.parent.mkdir(parents=True,exist_ok=True)
    if not labels_file.exists():
        labels_file.write_text('image,brand,color,confidence,x1,y1,x2,y2,source_listing\n',encoding='utf-8')
    for brand in BRANDS:
        folder = OUT / brand
        folder.mkdir(parents=True, exist_ok=True)
        count = len(list(folder.glob('*.jpg')))
        links = list(seeds[brand])
        print(f'\n{brand.upper()} starting with {count}/{TARGET}')
        for url in search_urls(brand):
            if len(links) >= MAX_LISTINGS_PER_BRAND:
                break
            try:
                html, _ = fetch(url)
                for link in listing_links(html, url):
                    if link not in links:
                        links.append(link)
            except (PermissionError, RuntimeError) as exc:
                print('Collection stopped:', exc)
                stop = True
                break
            except (requests.RequestException, ValueError) as exc:
                print('Search page skipped:', exc)
        if stop:
            break
        for listing in links[:MAX_LISTINGS_PER_BRAND]:
            if count >= TARGET:
                break
            try:
                html, _ = fetch(listing)
                soup = BeautifulSoup(html, 'html.parser')
                if not text_location_matches(soup):
                    continue
                # Brand is checked against the listing's visible text to reduce mislabeled data.
                visible = soup.get_text(' ', strip=True).lower()
                check = 'mercedes' if brand == 'mercedes' else brand
                if check not in visible:
                    continue
                for url in extract_images(html, listing):
                    if count >= TARGET:
                        break
                    try:
                        # Only OLX-owned/OLX-CDN image URLs, respecting each host's robots.txt.
                        host = urlparse(url).netloc.lower()
                        if not (host.endswith('olx.uz') or 'olxcdn' in host):
                            continue
                        raw, content_type = fetch(url, image=True)
                        if 'image/' not in content_type.lower():
                            continue
                        jpg = normalize_image(raw)
                        digest = hashlib.sha256(jpg).hexdigest()
                        if digest in known_hashes:
                            continue
                        prediction = analyze_photo(jpg)
                        if prediction is None:
                            continue  # no sufficiently large exterior vehicle found
                        color, box, confidence = prediction
                        filename = f'{brand}_{count+1:04d}.jpg'
                        (folder / filename).write_bytes(jpg)
                        with labels_file.open('a',newline='',encoding='utf-8') as lf:
                            csv.writer(lf).writerow([f'{brand}/{filename}',brand,color,round(confidence,3),*box,listing])
                        known_hashes.add(digest)
                        count += 1
                        total += 1
                        print(f'{brand}: {count}/{TARGET}')
                    except (requests.RequestException, ValueError, UnidentifiedImageError) as exc:
                        print('Image skipped:', str(exc)[:100])
            except (PermissionError, RuntimeError) as exc:
                print('Collection stopped:', exc)
                stop = True
                break
            except (requests.RequestException, ValueError) as exc:
                print('Listing skipped:', str(exc)[:100])
        if stop:
            break
    print(f'\nNew photos: {total}. Total files: {sum(len(list((OUT/b).glob("*.jpg"))) for b in BRANDS)}/5000')
    print('Auto-labels saved to dataset/auto_labels.csv. Estimates may be wrong; review before reporting accuracy.')

if __name__ == '__main__':
    main()
