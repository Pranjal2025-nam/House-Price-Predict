import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Royal Estates | Colony Plots & Homes", page_icon="🏡", layout="wide")

# ============================================================
#  MODEL (loaded from your pickles)
# ============================================================
@st.cache_resource
def load_artifacts():
    return joblib.load("house_model.pkl"), joblib.load("scaler.pkl"), joblib.load("columns.pkl")


model, scaler, COLS = load_artifacts()


def predict_lakh(area, beds, baths, balc, floors, age, parking, furnish, d_city, on_hw, d_hw, road_w):
    row = pd.DataFrame([{
        "area_sqft": area, "bedrooms": beds, "bathrooms": baths, "balconies": balc,
        "floors": floors, "age_years": age, "parking_slots": parking, "furnishing": furnish,
        "distance_city_center_km": d_city, "on_highway": on_hw,
        "distance_highway_km": d_hw, "road_width_ft": road_w,
    }])[COLS]
    return float(model.predict(scaler.transform(row))[0])


# ============================================================
#  COLONY CATALOG
# ============================================================
COLONIES = {
    "Green Valley Residency": {
        "tier": "Normal Colony", "emoji": "🌿", "color": "#2e7d32", "accent": "#a5d6a7",
        "tagline": "Affordable family living, away from the noise",
        "on_hw": 0, "d_hw": 3.0, "road_w": 30, "d_city": 12.0, "premium": 0.00,
        "plots": [900, 1200, 1500, 1800], "homes": [1000, 1400, 1800],
        "amenities": ["Kids play park", "Walking track", "24x7 security", "Community hall", "Street lights & CCTV"],
        "pool": False, "parks": 1, "scene": "colony",
    },
    "Highway Heights": {
        "tier": "Normal Colony", "emoji": "🛣️", "color": "#1565c0", "accent": "#90caf9",
        "tagline": "Right on the highway: shops, transport and fast access",
        "on_hw": 1, "d_hw": 0.1, "road_w": 80, "d_city": 9.0, "premium": 0.02,
        "plots": [1000, 1500, 2000, 2500], "homes": [1400, 1900, 2400],
        "amenities": ["Highway frontage", "Commercial strip", "2 kids play parks", "Bus & metro access", "Gated entry", "EV charging"],
        "pool": False, "parks": 2, "scene": "highway",
    },
    "Royal Palm Estate": {
        "tier": "Luxury Colony", "emoji": "🌴", "color": "#b8860b", "accent": "#ffe082",
        "tagline": "Resort-style villas with a private clubhouse",
        "on_hw": 1, "d_hw": 0.2, "road_w": 100, "d_city": 7.0, "premium": 0.08,
        "plots": [2000, 2800, 3500, 4000], "homes": [2400, 3000, 3600],
        "amenities": ["Luxury swimming pool", "3 kids play parks", "Clubhouse & gym", "Jogging track",
                      "Landscaped gardens", "Indoor games room", "24x7 security & CCTV", "Power backup"],
        "pool": True, "parks": 3, "scene": "luxury",
    },
    "Skyline Imperial Villas": {
        "tier": "Luxury Colony", "emoji": "👑", "color": "#6a1b9a", "accent": "#ce93d8",
        "tagline": "Ultra-premium gated villas near the city centre",
        "on_hw": 1, "d_hw": 0.1, "road_w": 120, "d_city": 3.5, "premium": 0.12,
        "plots": [2500, 3200, 3800, 4000], "homes": [3000, 3500, 4000],
        "amenities": ["Infinity swimming pool", "4 kids play parks", "Spa & sauna", "Tennis court", "Mini theatre",
                      "Sky lounge", "Smart-home wiring", "Concierge & valet", "EV charging"],
        "pool": True, "parks": 4, "scene": "luxury2",
    },
}

# ============================================================
#  ILLUSTRATIONS (original SVG artwork, no external images needed)
# ============================================================
def svg_wrap(inner, sky1, sky2):
    return f"""<svg viewBox='0 0 800 420' xmlns='http://www.w3.org/2000/svg' preserveAspectRatio='xMidYMid meet'>
<defs><linearGradient id='s{abs(hash(sky1+sky2))}' x1='0' y1='0' x2='0' y2='1'>
<stop offset='0' stop-color='{sky1}'/><stop offset='1' stop-color='{sky2}'/></linearGradient></defs>
<rect width='800' height='420' fill='url(#s{abs(hash(sky1+sky2))})'/><g fill='#fff' opacity='.85'><g><ellipse cx='0' cy='90' rx='60' ry='20'/><ellipse cx='40' cy='80' rx='40' ry='18'/><animateTransform attributeName='transform' type='translate' from='-150 0' to='950 0' dur='38s' repeatCount='indefinite'/></g><g><ellipse cx='0' cy='150' rx='50' ry='16'/><ellipse cx='34' cy='142' rx='32' ry='14'/><animateTransform attributeName='transform' type='translate' from='-300 0' to='900 0' dur='55s' repeatCount='indefinite'/></g></g>{inner}</svg>"""


def house(x, y, wall, roof, scale=1.0):
    s = scale
    return f"""<g transform='translate({x},{y}) scale({s})'>
<rect x='0' y='40' width='150' height='90' fill='{wall}'/>
<polygon points='-12,42 75,-8 162,42' fill='{roof}'/>
<rect x='60' y='80' width='30' height='50' fill='#5d4037'/>
<rect x='14' y='62' width='30' height='28' fill='#e3f2fd'/><rect x='106' y='62' width='30' height='28' fill='#e3f2fd'/></g>"""


def tree(x, y, s=1.0):
    return f"""<g transform='translate({x},{y}) scale({s})'><rect x='-4' y='20' width='8' height='30' fill='#6d4c41'/>
<circle cx='0' cy='10' r='24' fill='#2e7d32'/><circle cx='-14' cy='22' r='16' fill='#388e3c'/><circle cx='14' cy='22' r='16' fill='#43a047'/></g>"""


def scene_colony():
    return svg_wrap(
        "<circle cx='680' cy='70' r='38' fill='#ffeb3b'/><rect y='300' width='800' height='120' fill='#7cb342'/>"
        "<rect y='355' width='800' height='34' fill='#546e7a'/><rect x='0' y='370' width='800' height='3' fill='#eceff1'/>"
        + house(60, 200, "#ffe0b2", "#bf360c") + house(260, 205, "#c8e6c9", "#37474f", 0.95)
        + house(470, 200, "#bbdefb", "#6d4c41") + tree(650, 262, 1.2) + tree(20, 270, 0.9)
        + "",
        "#b3e5fc", "#e8f5e9")


def scene_highway():
    return svg_wrap(
        "<rect y='300' width='800' height='120' fill='#8bc34a'/><rect y='330' width='800' height='80' fill='#455a64'/>"
        "<g fill='#fff'><rect x='20' y='366' width='60' height='6'/><rect x='140' y='366' width='60' height='6'/>"
        "<rect x='260' y='366' width='60' height='6'/><rect x='380' y='366' width='60' height='6'/>"
        "<rect x='500' y='366' width='60' height='6'/><rect x='620' y='366' width='60' height='6'/></g>"
        + house(80, 190, "#e1f5fe", "#0d47a1") + house(300, 195, "#fff3e0", "#1565c0", 0.95)
        + "<rect x='540' y='175' width='190' height='130' fill='#cfd8dc'/><rect x='540' y='165' width='190' height='16' fill='#1565c0'/>"
        "<text x='635' y='178' text-anchor='middle' font-size='11' fill='#fff' font-family='Arial'>SHOPS &amp; MARKET</text>"
        "<rect x='160' y='338' width='90' height='30' rx='8' fill='#e53935'/><circle cx='185' cy='370' r='9' fill='#212121'/><circle cx='228' cy='370' r='9' fill='#212121'/>"
        "<rect x='520' y='345' width='100' height='28' rx='8' fill='#fdd835'/><circle cx='542' cy='374' r='9' fill='#212121'/><circle cx='598' cy='374' r='9' fill='#212121'/>"
        "",
        "#81d4fa", "#e1f5fe")


def scene_luxury():
    return svg_wrap(
        "<circle cx='90' cy='70' r='34' fill='#ffd54f'/><rect y='300' width='800' height='120' fill='#689f38'/>"
        "<rect x='60' y='150' width='430' height='150' fill='#fff8e1'/><rect x='40' y='135' width='470' height='22' fill='#b8860b'/>"
        "<rect x='110' y='185' width='50' height='60' fill='#90caf9'/><rect x='220' y='185' width='50' height='60' fill='#90caf9'/><rect x='330' y='185' width='50' height='60' fill='#90caf9'/>"
        "<rect x='420' y='215' width='38' height='85' fill='#6d4c41'/>"
        "<rect x='520' y='290' width='250' height='95' rx='18' fill='#26c6da'/><rect x='532' y='300' width='226' height='75' rx='12' fill='#4dd0e1'/>"
        "<path d='M545 335 q 20 -14 40 0 t 40 0 t 40 0 t 40 0 t 40 0' stroke='#e0f7fa' stroke-width='4' fill='none'/>"
        "<text x='645' y='282' text-anchor='middle' font-size='13' fill='#006064' font-weight='bold' font-family='Arial'>INFINITY SWIMMING POOL</text>"
        + tree(30, 250, 1.1) + tree(510, 250, 0.8)
        + "",
        "#ffe0b2", "#fff8e1")


def scene_luxury2():
    return svg_wrap(
        "<rect y='310' width='800' height='110' fill='#4a148c' opacity='.15'/>"
        "<rect y='310' width='800' height='110' fill='#558b2f'/>"
        "<rect x='90' y='90' width='70' height='220' fill='#ede7f6'/><rect x='170' y='60' width='80' height='250' fill='#d1c4e9'/>"
        "<rect x='260' y='110' width='70' height='200' fill='#ede7f6'/><rect x='340' y='140' width='150' height='170' fill='#f3e5f5'/>"
        "<g fill='#ffe082'><rect x='105' y='110' width='14' height='14'/><rect x='130' y='110' width='14' height='14'/><rect x='105' y='145' width='14' height='14'/><rect x='130' y='145' width='14' height='14'/>"
        "<rect x='190' y='85' width='14' height='14'/><rect x='215' y='85' width='14' height='14'/><rect x='190' y='120' width='14' height='14'/><rect x='215' y='120' width='14' height='14'/>"
        "<rect x='275' y='130' width='14' height='14'/><rect x='300' y='130' width='14' height='14'/></g>"
        "<rect x='520' y='300' width='250' height='90' rx='45' fill='#00acc1'/><rect x='534' y='310' width='222' height='70' rx='35' fill='#26c6da'/>"
        "<circle cx='700' cy='80' r='30' fill='#fff59d'/>"
        + tree(60, 262, 1.0) + tree(500, 262, 0.8)
        + "",
        "#d1c4e9", "#f3e5f5")


def scene_kids_park():
    return svg_wrap(
        "<rect y='290' width='800' height='130' fill='#8bc34a'/>"
        # swing set
        "<g stroke='#6d4c41' stroke-width='8'><line x1='80' y1='290' x2='140' y2='150'/><line x1='260' y1='290' x2='200' y2='150'/><line x1='130' y1='150' x2='210' y2='150'/></g>"
        "<g stroke='#555' stroke-width='3'><line x1='150' y1='150' x2='150' y2='250'/><line x1='190' y1='150' x2='190' y2='250'/></g>"
        "<rect x='135' y='250' width='30' height='8' fill='#e53935'/><rect x='175' y='250' width='30' height='8' fill='#fdd835'/>"
        # slide
        "<rect x='380' y='150' width='60' height='10' fill='#1e88e5'/><rect x='385' y='160' width='8' height='130' fill='#6d4c41'/><rect x='430' y='160' width='8' height='130' fill='#6d4c41'/>"
        "<polygon points='440,150 560,290 530,290 440,170' fill='#fb8c00'/>"
        # see-saw + sandpit
        "<polygon points='620,290 650,260 680,290' fill='#6d4c41'/><rect x='590' y='248' width='120' height='9' rx='4' fill='#e91e63' transform='rotate(-12 650 252)'/>"
        "<ellipse cx='300' cy='345' rx='70' ry='20' fill='#ffe0b2'/>"
        + tree(740, 240, 1.1) + tree(30, 250, 0.9)
        + "<g><circle cx='150' cy='235' r='9' fill='#ffcc80'/><rect x='144' y='244' width='12' height='14' fill='#42a5f5'/></g>"
        "",
        "#b3e5fc", "#f1f8e9")


def scene_pool():
    return svg_wrap(
        "<rect y='250' width='800' height='170' fill='#d7ccc8'/>"
        "<rect x='90' y='270' width='620' height='120' rx='30' fill='#00acc1'/><rect x='108' y='284' width='584' height='92' rx='22' fill='#4dd0e1'/>"
        "<path d='M130 330 q 30 -18 60 0 t 60 0 t 60 0 t 60 0 t 60 0 t 60 0 t 60 0 t 60 0 t 60 0' stroke='#e0f7fa' stroke-width='5' fill='none'/>"
        # palms + loungers
        "<g stroke='#6d4c41' stroke-width='9'><line x1='50' y1='260' x2='60' y2='130'/><line x1='750' y1='260' x2='740' y2='130'/></g>"
        "<g fill='#2e7d32'><ellipse cx='60' cy='125' rx='55' ry='14' transform='rotate(-20 60 125)'/><ellipse cx='60' cy='125' rx='55' ry='14' transform='rotate(25 60 125)'/>"
        "<ellipse cx='740' cy='125' rx='55' ry='14' transform='rotate(20 740 125)'/><ellipse cx='740' cy='125' rx='55' ry='14' transform='rotate(-25 740 125)'/></g>"
        "<rect x='140' y='232' width='70' height='14' rx='6' fill='#fff'/><rect x='580' y='232' width='70' height='14' rx='6' fill='#fff'/>"
        "<circle cx='400' y='90' cy='80' r='38' fill='#ffd54f'/>"
        "",
        "#ffe0b2", "#fff3e0")


def scene_clubhouse():
    return svg_wrap(
        "<rect y='300' width='800' height='120' fill='#7cb342'/>"
        "<rect x='150' y='140' width='500' height='160' fill='#f5f5f5'/><polygon points='130,142 400,60 670,142' fill='#37474f'/>"
        "<g fill='#90caf9'><rect x='190' y='180' width='60' height='70'/><rect x='300' y='180' width='60' height='70'/><rect x='440' y='180' width='60' height='70'/><rect x='550' y='180' width='60' height='70'/></g>"
        "<rect x='370' y='210' width='60' height='90' fill='#8d6e63'/>"
        "<circle cx='400' cy='112' r='14' fill='#ffd54f'/>"
        + tree(80, 255, 1.2) + tree(730, 255, 1.2)
        + "",
        "#cfd8dc", "#eceff1")


def scene_plot():
    return svg_wrap(
        "<rect y='260' width='800' height='160' fill='#8bc34a'/>"
        "<polygon points='60,420 740,420 640,290 160,290' fill='#c5e1a5' stroke='#fff' stroke-width='4' stroke-dasharray='14 8'/>"
        "<g fill='#9e9e9e'><rect x='150' y='276' width='14' height='26'/><rect x='636' y='276' width='14' height='26'/>"
        "<rect x='40' y='404' width='16' height='30'/><rect x='744' y='404' width='16' height='30'/></g>"
        "<rect x='360' y='190' width='6' height='110' fill='#6d4c41'/><rect x='300' y='150' width='130' height='52' rx='6' fill='#fff'/>"
        "<text x='365' y='183' text-anchor='middle' font-size='20' font-family='Arial' font-weight='bold' fill='#c62828'>PLOT FOR SALE</text>"
        + tree(90, 230, 1.0) + tree(710, 232, 1.0),
        "#b3e5fc", "#f1f8e9")


def _room(items, wall="#efe6d8", floor="#b08968"):
    def f():
        return svg_wrap(
            f"<rect width='800' height='420' fill='{wall}'/><rect y='300' width='800' height='120' fill='{floor}'/>" + items,
            wall, wall)
    return f


scene_hall = _room("<rect x='520' y='70' width='200' height='150' fill='#bbdefb' stroke='#fff' stroke-width='8'/>"
                   "<rect x='90' y='210' width='330' height='100' rx='18' fill='#5d7a8c'/><rect x='80' y='240' width='40' height='90' rx='14' fill='#4a6577'/>"
                   "<rect x='390' y='240' width='40' height='90' rx='14' fill='#4a6577'/><ellipse cx='260' cy='355' rx='170' ry='28' fill='#c9a66b'/>"
                   "<rect x='590' y='250' width='6' height='80' fill='#333'/><polygon points='570,250 616,250 604,215 582,215' fill='#ffd54f'/>")
scene_bedroom = _room("<rect x='90' y='120' width='330' height='60' rx='10' fill='#8d6e63'/><rect x='80' y='190' width='350' height='130' rx='16' fill='#f5f5f5'/>"
                      "<rect x='80' y='250' width='350' height='70' rx='12' fill='#90a4ae'/><rect x='110' y='200' width='90' height='40' rx='12' fill='#fff'/>"
                      "<rect x='450' y='250' width='70' height='70' fill='#6d4c41'/><polygon points='465,250 505,250 497,215 473,215' fill='#ffd54f'/>"
                      "<rect x='580' y='70' width='160' height='140' fill='#bbdefb' stroke='#fff' stroke-width='8'/>", "#e8eaf6", "#a1887f")
scene_kitchen = _room("<rect x='40' y='80' width='720' height='70' fill='#8d6e63'/><rect x='40' y='230' width='720' height='90' fill='#fafafa'/>"
                      "<rect x='40' y='222' width='720' height='12' fill='#546e7a'/><rect x='250' y='300' width='300' height='46' fill='#fff' stroke='#cfd8dc'/>"
                      "<g stroke='#555' stroke-width='3'><line x1='200' y1='20' x2='200' y2='70'/><line x1='400' y1='20' x2='400' y2='70'/><line x1='600' y1='20' x2='600' y2='70'/></g>"
                      "<g fill='#ffd54f'><circle cx='200' cy='78' r='16'/><circle cx='400' cy='78' r='16'/><circle cx='600' cy='78' r='16'/></g>", "#eceff1", "#90a4ae")
scene_dining = _room("<g stroke='#555' stroke-width='3'><line x1='400' y1='0' x2='400' y2='90'/></g><ellipse cx='400' cy='100' rx='60' ry='22' fill='#ffd54f'/>"
                     "<rect x='220' y='230' width='360' height='24' rx='6' fill='#6d4c41'/><rect x='250' y='254' width='12' height='90' fill='#6d4c41'/><rect x='540' y='254' width='12' height='90' fill='#6d4c41'/>"
                     "<rect x='200' y='250' width='46' height='90' rx='8' fill='#a1887f'/><rect x='554' y='250' width='46' height='90' rx='8' fill='#a1887f'/>"
                     "<rect x='80' y='60' width='150' height='110' fill='#bbdefb' stroke='#fff' stroke-width='8'/>")
scene_furniture = _room("<rect x='70' y='215' width='360' height='110' rx='20' fill='#7b5e57'/><rect x='60' y='245' width='44' height='90' rx='16' fill='#654b45'/>"
                        "<rect x='396' y='245' width='44' height='90' rx='16' fill='#654b45'/><rect x='500' y='110' width='220' height='220' fill='#8d6e63'/>"
                        "<g fill='#efebe9'><rect x='515' y='125' width='190' height='50'/><rect x='515' y='190' width='190' height='50'/><rect x='515' y='255' width='190' height='55'/></g>"
                        "<rect x='140' y='330' width='220' height='16' rx='4' fill='#4e342e'/>")
scene_bathroom = _room("<rect x='90' y='230' width='330' height='90' rx='40' fill='#fff' stroke='#cfd8dc' stroke-width='4'/>"
                       "<rect x='520' y='80' width='160' height='170' rx='12' fill='#e3f2fd' stroke='#90a4ae' stroke-width='8'/>"
                       "<rect x='500' y='260' width='200' height='50' rx='8' fill='#8d6e63'/><ellipse cx='600' cy='262' rx='50' ry='14' fill='#fff'/>", "#e0f2f1", "#b0bec5")


SCENES = {
    "colony": scene_colony, "highway": scene_highway, "luxury": scene_luxury, "luxury2": scene_luxury2,
    "kids": scene_kids_park, "pool": scene_pool, "club": scene_clubhouse,
    "plot": scene_plot, "home": scene_colony, "park": scene_kids_park, "hall": scene_hall, "bedroom": scene_bedroom,
    "kitchen": scene_kitchen, "dining": scene_dining, "furniture": scene_furniture, "bathroom": scene_bathroom,
}



# ============================================================
#  MEDIA: real photos from ./images (if present) else drawings
# ============================================================
import base64
import glob
import os

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
CAPTIONS = {
    "colony": "Green Valley Residency", "highway": "Highway Heights", "luxury": "Royal Palm Estate",
    "luxury2": "Skyline Imperial Villas", "kids": "Kids Play Park", "pool": "Luxury Swimming Pool",
    "club": "Clubhouse & Gym", "park": "Colony Garden Park", "plot": "Residential Plots", "home": "Ready Homes", "hall": "Living Hall", "bedroom": "Master Bedroom",
    "kitchen": "Modular Kitchen", "dining": "Dining Area", "furniture": "Premium Furniture", "bathroom": "Designer Bathroom",
}

import json
import re

CUSTOM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_colonies.json")


def load_custom():
    try:
        with open(CUSTOM_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_custom(data):
    with open(CUSTOM_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


CUSTOM = load_custom()
for _name, _c in CUSTOM.items():          # your own colonies, added from the "Add Colony" tab
    COLONIES[_name] = _c
    CAPTIONS[_c["scene"]] = _name
    SCENES[_c["scene"]] = SCENES["luxury" if _c.get("pool") else "colony"]


@st.cache_data
def photo_uris(key):
    uris = []
    for ext in ("jpg", "jpeg", "png", "webp"):
        for f in sorted(glob.glob(os.path.join(IMG_DIR, f"{key}_*.{ext}")) + glob.glob(os.path.join(IMG_DIR, f"{key}.{ext}"))):
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            with open(f, "rb") as fh:
                uris.append(f"data:image/{mime};base64," + base64.b64encode(fh.read()).decode())
    return uris


def count_real_photos():
    return sum(len(photo_uris(k)) for k in CAPTIONS)


def _u(photo_id):
    return f"https://images.unsplash.com/photo-{photo_id}?auto=format&fit=crop&w=1400&q=70"


# Real photos from Unsplash (free to use). Your own photos in ./images take priority.
REMOTE_PHOTOS = {
    "colony":  [_u("1631562271453-dd01b12a8141"), _u("1605640214887-5f1f80e46cd4"), _u("1572120360610-d971b9d7767c"), _u("1596573513513-d56b3999ac38")],
    "park":    [_u("1695551407214-25a5dc6300b8"), _u("1710689208379-2b2f1c6620ca"), _u("1718206798129-b9b9c3ec48a3"), _u("1632214921412-545c05252059")],
    "highway": [_u("1543996991-8e851c2dc841"), _u("1563242152-568e5de6f2b8")],
    "luxury":  [_u("1613490493576-7fde63acd811"), _u("1505843513577-22bb7d21e455")],
    "luxury2": [_u("1624204386084-dd8c05e32226"), _u("1706808849780-7a04fbac83ef"), _u("1670589953882-b94c9cb380f5")],
    "pool":    [_u("1603034203013-d532350372c6"), _u("1711114378532-bab7d55311f3"), _u("1721222204632-bf9abe6f023f")],
    "kids":    [_u("1777626759287-1b0980543104"), _u("1759776050712-2b7880af8cd4"), _u("1758445516786-4980d65f6c04")],
    "club":    [_u("1761971976282-b2bb051a5474")],
    "plot":    [_u("1773215023063-e662ea91a69c"), _u("1669003152631-c953ac9f3ed3"), _u("1494187570835-b188e7f0f26e"), _u("1495107334309-fcf20504a5ab")],
    "hall":      [_u("1583847268964-b28dc8f51f92"), _u("1554995207-c18c203602cb"), _u("1649083048770-82e8ffd80431"), _u("1616047006789-b7af5afb8c20"), _u("1613545325278-f24b0cae1224")],
    "bedroom":   [_u("1616594039964-ae9021a400a0"), _u("1778731525620-ae459167c176"), _u("1615874959474-d609969a20ed"), _u("1616486029423-aaa4789e8c9a"), _u("1768487422639-7ba3900d0f02")],
    "kitchen":   [_u("1556911220-bff31c812dba"), _u("1600489000022-c2086d79f9d4"), _u("1617228069096-4638a7ffc906"), _u("1484154218962-a197022b5858"), _u("1622372738946-62e02505feb3")],
    "dining":    [_u("1616486886892-ff366aa67ba4"), _u("1617806118233-18e1de247200"), _u("1773578978637-c9771e7a9913"), _u("1604578762246-41134e37f9cc")],
    "furniture": [_u("1555041469-a586c61ea9bc"), _u("1605774337664-7a846e9cdf17"), _u("1564078516393-cf04bd966897"), _u("1634712282287-14ed57b9cc89"), _u("1691480152351-4b3f2c89ccff")],
    "bathroom":  [_u("1584622650111-993a426fbf0a"), _u("1763485956310-55f3c0e822d5"), _u("1631889993959-41b4e9c6e3c5"), _u("1620626011761-996317b8d101")],
    "home":    [_u("1613977257363-707ba9348227"), _u("1758448756207-54505680d130"), _u("1721815693498-cc28507c0ba2"), _u("1748063578185-3d68121b11ff"), _u("1554995207-c18c203602cb")],
}


STOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_hidden.json")


def load_hidden():
    try:
        with open(STOCK_FILE, encoding="utf-8") as fh:
            return set(json.load(fh))
    except Exception:
        return set()


HIDDEN = load_hidden()


def media_slides(keys):
    out = []
    for k in keys:
        local = photo_uris(k)
        if local:  # your own photos
            out += [(f"<img src='{u}'>", CAPTIONS[k]) for u in local]
            continue
        fallback = SCENES[k]().replace("xMidYMid meet", "xMidYMid slice")
        for url in [u for u in REMOTE_PHOTOS.get(k, []) if u not in HIDDEN]:
            out.append((f"<img src='{url}' referrerpolicy='no-referrer' "
                        "onerror=\"this.style.display='none';this.nextElementSibling.style.display='block'\">"
                        f"<div class='fb' style='display:none;height:100%'>{fallback}</div>", CAPTIONS[k]))
        if k not in REMOTE_PHOTOS:       # your own colony with no photos yet
            out.append((fallback, CAPTIONS[k]))
    if not out and keys:                 # everything hidden: never leave the slider empty
        out.append((SCENES[keys[0]]().replace("xMidYMid meet", "xMidYMid slice"), CAPTIONS[keys[0]]))
    return out


def price_html(label, lakh, area):
    return f"""
<style>
 body{{margin:0;font-family:'Segoe UI',Arial,sans-serif;background:transparent}}
 .p{{position:relative;overflow:hidden;text-align:center;padding:22px 10px;border-radius:22px;color:#201500;
     background:linear-gradient(120deg,#b8860b,#f3d36b 60%,#fff2b3);animation:glow 3s infinite,pop .8s both}}
 .p::after{{content:'';position:absolute;top:0;left:0;width:40%;height:100%;
     background:linear-gradient(100deg,transparent,#ffffff99,transparent);animation:sh 3.2s 1.2s infinite}}
 h1{{margin:6px 0;font-size:3.2rem;font-family:Georgia,serif}} .s{{font-size:1.02rem;font-weight:600}}
 @keyframes sh{{0%{{transform:translateX(-120%)}}100%{{transform:translateX(260%)}}}}
 @keyframes pop{{from{{opacity:0;transform:scale(.92)}}to{{opacity:1;transform:none}}}}
 @keyframes glow{{0%,100%{{box-shadow:0 10px 30px #d4af3755}}50%{{box-shadow:0 10px 52px #f3d36bcc}}}}
</style>
<div class='p'><div>{label}</div><h1>&#8377; <span id='v'>0.0</span> Lakh</h1>
<div class='s' id='sub'></div></div>
<script>
 var to={lakh:.4f}, area={area}, t0=null, v=document.getElementById('v'), sub=document.getElementById('sub');
 function fmt(x){{return x.toLocaleString('en-IN',{{minimumFractionDigits:1,maximumFractionDigits:1}})}}
 function step(ts){{ if(!t0)t0=ts; var p=Math.min((ts-t0)/1600,1), e=1-Math.pow(1-p,3), x=to*e; v.textContent=fmt(x);
   sub.innerHTML='&asymp; &#8377; '+(x/100).toFixed(2)+' Crore &nbsp;|&nbsp; &#8377; '+Math.round(x*100000/area).toLocaleString('en-IN')+' per sqft';
   if(p<1)requestAnimationFrame(step); }}
 requestAnimationFrame(step);
</script>"""


def slider_html(keys, height=300, overlay="", captions=True, interval=4000):
    slides = media_slides(keys)
    body = "".join(
        f"<div class='slide'>{m}{'<div class=cap>' + c + '</div>' if captions else ''}</div>" for m, c in slides
    )
    n = len(slides)
    dots = "".join(f"<span class='dot' onclick='go({i})'></span>" for i in range(n))
    return f"""
<style>
 *{{box-sizing:border-box}} body{{margin:0;padding:0;font-family:'Segoe UI',Arial,sans-serif;background:transparent}}
 .wrap{{position:relative;overflow:hidden;border-radius:20px;height:{height}px;box-shadow:0 12px 34px rgba(0,0,0,.55);border:1px solid #d4af3744}}
 .track{{display:flex;height:100%;transition:transform .9s cubic-bezier(.7,0,.2,1)}}
 .slide{{min-width:100%;height:100%;position:relative;overflow:hidden}}
 .slide img,.slide svg{{animation:kb 10s ease-in-out infinite alternate}}
 @keyframes kb{{from{{transform:scale(1)}}to{{transform:scale(1.13) translate(-1.5%,-1%)}}}}
 @keyframes up{{from{{opacity:0;transform:translateY(26px)}}to{{opacity:1;transform:none}}}}
 @keyframes pulse{{0%,100%{{box-shadow:0 0 0 0 #f3d36b88}}50%{{box-shadow:0 0 0 14px #f3d36b00}}}}
 .ov>*{{animation:up .9s both}} .ov>*:nth-child(2){{animation-delay:.25s}} .ov>*:nth-child(3){{animation-delay:.5s;animation:up .9s .5s both,pulse 2.2s 1.5s infinite}}
 .cap{{animation:up .8s .3s both}}
 .slide img,.slide svg{{width:100%;height:100%;object-fit:cover;display:block}}
 .slide::after{{content:'';position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,0,0,.05) 40%,rgba(5,10,25,.78));pointer-events:none}}
 .cap{{position:absolute;left:22px;bottom:34px;z-index:2;color:#fff;font-size:1.05rem;font-weight:600;letter-spacing:.4px;
       background:#0009;padding:5px 14px;border-radius:20px;border:1px solid #d4af3799}}
 .ov{{position:absolute;inset:0;z-index:3;display:flex;flex-direction:column;justify-content:center;padding:0 46px;pointer-events:none}}
 .btn{{position:absolute;top:50%;transform:translateY(-50%);z-index:4;background:#000a;color:#d4af37;border:1px solid #d4af37;
      width:42px;height:42px;border-radius:50%;font-size:20px;cursor:pointer}}
 .btn:hover{{background:#d4af37;color:#000}} .prev{{left:12px}} .next{{right:12px}}
 .dots{{position:absolute;bottom:10px;width:100%;text-align:center;z-index:4}}
 .dot{{display:inline-block;width:9px;height:9px;margin:0 4px;border-radius:50%;background:#fff6;cursor:pointer}}
 .dot.on{{background:#d4af37;width:24px;border-radius:6px;transition:.3s}}
</style>
<div class='wrap'><div class='track' id='t'>{body}</div>{overlay}
<button class='btn prev' onclick='go(i-1)'>&#10094;</button><button class='btn next' onclick='go(i+1)'>&#10095;</button>
<div class='dots'>{dots}</div></div>
<script>
 var i=0,n={n},t=document.getElementById('t'),ds=document.querySelectorAll('.dot'),timer;
 function go(k){{i=(k+n)%n;t.style.transform='translateX(-'+(i*100)+'%)';ds.forEach((d,x)=>d.classList.toggle('on',x==i));
   clearInterval(timer);timer=setInterval(()=>go(i+1),{interval});}}
 go(0);
</script>"""


# ============================================================
#  STYLE (dark luxury)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;800&family=Poppins:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'Poppins',sans-serif}
.stApp{background:radial-gradient(1200px 600px at 10% -10%,#1c2a52 0%,transparent 60%),
       radial-gradient(900px 500px at 100% 0%,#3a2c0a 0%,transparent 55%),#0a1020}
.block-container{padding-top:1.4rem;max-width:1180px}
h1,h2,h3{font-family:'Playfair Display',serif!important;color:#f4efe3}
.gold{color:#d4af37}
.hero-title{font-family:'Playfair Display',serif;font-size:3rem;font-weight:800;color:#fff;line-height:1.1;text-shadow:0 3px 18px #000}
.hero-sub{color:#f4efe3;font-size:1.1rem;margin-top:8px;text-shadow:0 2px 10px #000;max-width:560px}
.hero-btn{display:inline-block;margin-top:16px;background:linear-gradient(120deg,#b8860b,#f3d36b);color:#201500;font-weight:700;
          padding:10px 22px;border-radius:30px;width:max-content}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0 8px}
.stat{background:linear-gradient(160deg,#18223d,#0f1730);border:1px solid #d4af3733;border-radius:16px;padding:16px;text-align:center;
      box-shadow:0 6px 20px #0006}
.stat b{display:block;font-family:'Playfair Display',serif;font-size:1.9rem;color:#d4af37}
.stat span{font-size:.82rem;color:#c9c4b6;letter-spacing:.5px;text-transform:uppercase}
.card{background:linear-gradient(160deg,#18223d,#0f1730);border:1px solid #d4af3733;border-radius:20px;padding:20px 22px;
      box-shadow:0 10px 30px #0007;transition:.3s}
.card:hover{transform:translateY(-4px);border-color:#d4af37aa}
.card h3{margin:6px 0 2px;font-size:1.5rem}
.tag{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.72rem;font-weight:700;letter-spacing:.6px;margin-right:6px;text-transform:uppercase}
.tag.lux{background:linear-gradient(120deg,#b8860b,#f3d36b);color:#201500}
.tag.real{background:#1e88e5;color:#fff} .tag.hw{background:#37474f;color:#fff} .tag.in{background:#2e7d32;color:#fff}
.row{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}
.pill{background:#ffffff12;border:1px solid #ffffff22;border-radius:12px;padding:6px 12px;font-size:.85rem}
.pill b{color:#d4af37}
.chip{display:inline-block;background:#d4af3722;color:#f3d36b;border:1px solid #d4af3755;border-radius:14px;padding:3px 11px;margin:3px 3px 0 0;font-size:.78rem}
.price{background:linear-gradient(120deg,#b8860b,#f3d36b 60%,#fff2b3);color:#201500;border-radius:22px;padding:26px;text-align:center;
       box-shadow:0 12px 36px #d4af3755}
.price h1{margin:4px 0;font-size:3.1rem;color:#201500!important}
.mini{background:#131b30;border:1px solid #d4af3733;border-radius:14px;padding:12px;text-align:center}
.mini b{display:block;color:#d4af37;font-size:1.3rem}
.feature{background:#131b30;border:1px solid #d4af3722;border-radius:16px;padding:16px;height:100%}
.feature h4{margin:0 0 6px;color:#f3d36b}
[data-testid="stSidebar"]{border-right:1px solid #d4af3733}
.stTabs [data-baseweb="tab"]{font-weight:600}
@keyframes fadeUp{from{opacity:0;transform:translateY(34px)}to{opacity:1;transform:none}}
@keyframes shimmer{0%{transform:translateX(-120%)}100%{transform:translateX(220%)}}
@keyframes glow{0%,100%{box-shadow:0 12px 36px #d4af3755}50%{box-shadow:0 12px 56px #f3d36bcc}}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
@property --n{syntax:'<integer>';inherits:false;initial-value:0}
@keyframes count{from{--n:0}to{--n:var(--to)}}
.stat{animation:fadeUp .8s both} .stat:nth-child(2){animation-delay:.12s} .stat:nth-child(3){animation-delay:.24s} .stat:nth-child(4){animation-delay:.36s}
.stat:hover{transform:translateY(-6px);transition:.3s;border-color:#d4af37}
.stat b.cnt{animation:count 2s ease-out both;counter-reset:num var(--n)} .stat b.cnt::after{content:counter(num)}
.card{animation:fadeUp .9s both} .feature{animation:fadeUp .8s both;transition:.3s} .feature:hover{transform:translateY(-5px);border-color:#d4af37aa}
.mini{animation:fadeUp .7s both;transition:.3s} .mini:hover{transform:scale(1.05)}
.price{position:relative;overflow:hidden;animation:fadeUp .8s both,glow 3s 1s infinite}
.price::after{content:'';position:absolute;top:0;left:0;width:40%;height:100%;
  background:linear-gradient(100deg,transparent,#ffffff88,transparent);animation:shimmer 3.2s 1s infinite}
.tag.lux{animation:floaty 2.6s ease-in-out infinite}
.chip{transition:.25s} .chip:hover{background:#d4af37;color:#201500;transform:translateY(-2px)}
.pill{transition:.25s} .pill:hover{background:#d4af3733;transform:translateY(-2px)}
.footer{text-align:center;color:#8d8a80;padding:26px 0 8px;font-size:.85rem}
</style>""", unsafe_allow_html=True)


def fmt_money(lakh):
    return f"₹ {lakh / 100:,.2f} Cr" if lakh >= 100 else f"₹ {lakh:,.1f} L"


# ============================================================
#  HERO
# ============================================================
hero_overlay = """<div class='ov'><div style="font-family:'Playfair Display',Georgia,serif;font-size:2.9rem;font-weight:800;color:#fff;line-height:1.1;text-shadow:0 3px 18px #000">
Live where life feels <span style='color:#f3d36b'>royal</span></div>
<div style="color:#f4efe3;font-size:1.1rem;margin-top:8px;text-shadow:0 2px 10px #000;max-width:520px">
Premium plots &amp; ready homes · Swimming pools · Kids' parks · Highway-side &amp; inside colonies</div>
<div style="margin-top:16px;background:linear-gradient(120deg,#b8860b,#f3d36b);color:#201500;font-weight:700;padding:10px 22px;border-radius:30px;width:max-content">
Get instant price estimate &#8594;</div></div>"""
st.iframe(slider_html(["luxury", "luxury2", "pool", "highway", "colony", "kids", "club"], height=430,
                      overlay=hero_overlay, captions=False, interval=4500), height=450)

_all_prices = []
st.markdown(f"""<div class='stats'>
<div class='stat'><b class='cnt' style='--to:{len(COLONIES)}'></b><span>Colonies</span></div>
<div class='stat'><b class='cnt' style='--to:{sum(1 for v in COLONIES.values() if v['pool'])}'></b><span>Swimming pools</span></div>
<div class='stat'><b class='cnt' style='--to:{sum(v['parks'] for v in COLONIES.values())}'></b><span>Kids' parks</span></div>
<div class='stat'><b>900 - 4000</b><span>Sq ft plots &amp; homes</span></div></div>""", unsafe_allow_html=True)

# ============================================================
#  SIDEBAR
# ============================================================
st.sidebar.markdown("## 🔎 Find your colony")
tier = st.sidebar.radio("Colony type", ["All", "Normal Colony", "Luxury Colony"])
want_pool = st.sidebar.checkbox("🏊 Must have swimming pool")
min_parks = st.sidebar.slider("🛝 Minimum kids' parks", 0, 4, 0)
loc_pref = st.sidebar.radio("📍 Location", ["Any", "On highway", "Inside (off highway)"])
st.sidebar.divider()
nreal = count_real_photos()
if nreal:
    st.sidebar.success(f"📸 Using {nreal} real photo(s) from the images folder")
else:
    st.sidebar.info("📸 Showing Unsplash photos. To use your own project photos, add them to the **images** folder (see images/README.txt).")

visible = {
    k: v for k, v in COLONIES.items()
    if (tier == "All" or v["tier"] == tier) and (not want_pool or v["pool"]) and v["parks"] >= min_parks
    and (loc_pref == "Any" or (loc_pref == "On highway") == bool(v["on_hw"]))
}

tab1, tab2, tab3, tab_int, tab4, tab5, tab6 = st.tabs(
    ["🏘️ Colonies", "💰 Price & EMI", "🏊 Pool & Kids' Parks", "🛋️ Interiors", "🖼️ Gallery", "📞 Book a Visit", "➕ Add Colony / Photos"])

# ============================================================
#  TAB 1: COLONIES (2-column cards, photo first)
# ============================================================
with tab1:
    if not visible:
        st.warning("No colony matches these filters. Change the sidebar options.")
    items = list(visible.items())
    for r in range(0, len(items), 2):
        cols = st.columns(2, gap="large")
        for col, (name, c) in zip(cols, items[r:r + 2]):
            with col:
                st.iframe(slider_html([c["scene"], "park", "kids"] + (["pool", "club"] if c["pool"] else []), height=240), height=260)
                lux = c["tier"] == "Luxury Colony"
                chips = "".join(f"<span class='chip'>{a}</span>" for a in c["amenities"])
                low = predict_lakh(c["homes"][0], 2, 2, 1, 1, 0, 1, 1, c["d_city"], c["on_hw"], c["d_hw"], c["road_w"]) * (1 + c["premium"])
                st.markdown(f"""<div class='card'>
<span class='tag {"lux" if lux else "real"}'>{"★ Luxury" if lux else "Normal"}</span>
<span class='tag {"hw" if c["on_hw"] else "in"}'>{"On highway" if c["on_hw"] else "Inside colony"}</span>
<h3>{c['emoji']} {name}</h3><div style='color:#c9c4b6'><i>{c['tagline']}</i></div>
<div class='row'>
<div class='pill'>🏞️ Plots <b>{c['plots'][0]:,} - {c['plots'][-1]:,} sqft</b></div>
<div class='pill'>🏠 Homes <b>{c['homes'][0]:,} - {c['homes'][-1]:,} sqft</b></div></div>
<div class='row'>
<div class='pill'>🏊 Pool <b>{"Yes" if c["pool"] else "No"}</b></div>
<div class='pill'>🛝 Kids' parks <b>{c['parks']}</b></div>
<div class='pill'>📍 City <b>{c['d_city']} km</b></div>
<div class='pill'>🛣️ Road <b>{c['road_w']} ft</b></div></div>
<div style='margin:6px 0'>Homes from <b class='gold' style='font-size:1.3rem'>{fmt_money(low)}</b></div>
{chips}</div><br>""", unsafe_allow_html=True)

# ============================================================
#  TAB 2: PRICE + EMI
# ============================================================
with tab2:
    st.subheader("Get an instant price estimate")
    names = list(visible.keys()) or list(COLONIES.keys())
    a, b = st.columns(2)
    with a:
        cname = st.selectbox("Choose colony", names)
        c = COLONIES[cname]
        kind = st.radio("What do you want?", ["🏠 Ready Home", "🏞️ Plot"], horizontal=True)
    with b:
        sizes = c["homes"] if kind.startswith("🏠") else c["plots"]
        area = st.select_slider("Size (square feet)", options=sizes, value=sizes[len(sizes) // 2])
        st.caption(f"≈ {area / 9:,.0f} sq yards  •  {area * 0.0929:,.0f} sq metres")

    LAND_FACTOR = 0.55
    if kind.startswith("🏠"):
        h1, h2, h3 = st.columns(3)
        beds = h1.number_input("Bedrooms", 1, 6, max(1, min(6, round(area / 700))))
        baths = h2.number_input("Bathrooms", 1, 5, max(1, min(5, round(area / 900))))
        floors = h3.number_input("Floors", 1, 3, 2 if area > 1800 else 1)
        h4, h5, h6 = st.columns(3)
        balc = h4.number_input("Balconies", 0, 3, 1)
        parking = h5.number_input("Parking slots", 0, 3, 2)
        furn = h6.selectbox("Furnishing", [0, 1, 2], index=1,
                            format_func=lambda x: ["Unfurnished", "Semi-furnished", "Fully furnished"][x])
        age = st.slider("House age (years)", 0, 30, 0)
        base = predict_lakh(area, beds, baths, balc, floors, age, parking, furn, c["d_city"], c["on_hw"], c["d_hw"], c["road_w"])
        note = "Estimated by your Decision Tree model."
    else:
        base = predict_lakh(area, 1, 1, 0, 1, 0, 0, 0, c["d_city"], c["on_hw"], c["d_hw"], c["road_w"]) * LAND_FACTOR
        note = f"Plot price = model estimate for a bare-land profile x {LAND_FACTOR} (assumed land-only factor)."

    final = base * (1 + c["premium"])
    st.iframe(slider_html(["plot"] if not kind.startswith("🏠") else ["home", "hall", "bedroom", "kitchen"], height=250, interval=3000), height=270)
    st.iframe(price_html(f"Estimated price · {kind[2:]} · {cname}", final, area), height=215)
    m = st.columns(4)
    for col, (val, lab) in zip(m, [(f"{area:,}", "sq ft"), ("Yes" if c["pool"] else "No", "swimming pool"),
                                   (c["parks"], "kids' parks"), (f"+{c['premium'] * 100:.0f}%", "amenity premium")]):
        col.markdown(f"<div class='mini'><b>{val}</b>{lab}</div>", unsafe_allow_html=True)
    st.caption(f"{note} A colony amenity premium of {c['premium'] * 100:.0f}% is added for pool/parks/clubhouse "
               "because they are not features in the trained model. Estimates are indicative only.")

    st.markdown("### 🏦 EMI calculator")
    e1, e2, e3 = st.columns(3)
    down = e1.slider("Down payment (%)", 5, 60, 20)
    rate = e2.slider("Interest rate (% / year)", 6.0, 14.0, 8.75, 0.05)
    yrs = e3.slider("Loan tenure (years)", 5, 30, 20)
    loan = final * 1e5 * (1 - down / 100)
    r_m, n_m = rate / 1200, yrs * 12
    emi = loan * r_m * (1 + r_m) ** n_m / ((1 + r_m) ** n_m - 1)
    q = st.columns(3)
    q[0].markdown(f"<div class='mini'><b>₹ {emi:,.0f}</b>monthly EMI</div>", unsafe_allow_html=True)
    q[1].markdown(f"<div class='mini'><b>₹ {loan / 1e5:,.1f} L</b>loan amount</div>", unsafe_allow_html=True)
    q[2].markdown(f"<div class='mini'><b>₹ {(emi * n_m - loan) / 1e5:,.1f} L</b>total interest</div>", unsafe_allow_html=True)

    st.markdown("### 📊 Compare all colonies (same size)")
    rows = []
    for n_, cc in COLONIES.items():
        if kind.startswith("🏠"):
            p = predict_lakh(area, beds, baths, balc, floors, age, parking, furn, cc["d_city"], cc["on_hw"], cc["d_hw"], cc["road_w"])
        else:
            p = predict_lakh(area, 1, 1, 0, 1, 0, 0, 0, cc["d_city"], cc["on_hw"], cc["d_hw"], cc["road_w"]) * LAND_FACTOR
        rows.append({"Colony": n_, "Type": cc["tier"], "Price (lakh)": round(p * (1 + cc["premium"]), 1)})
    cmp_df = pd.DataFrame(rows).set_index("Colony")
    st.bar_chart(cmp_df["Price (lakh)"], color="#d4af37")
    st.dataframe(cmp_df, width="stretch")

# ============================================================
#  TAB 3: POOL & PARKS
# ============================================================
with tab3:
    st.subheader("🏊 Luxury swimming pool")
    p1, p2 = st.columns([1.5, 1], gap="large")
    with p1:
        st.iframe(slider_html(["pool", "luxury", "luxury2"], height=320), height=340)
    with p2:
        for t_, d_ in [("Infinity pool", "Temperature-controlled, with a separate shallow kids' pool"),
                       ("Poolside lounge", "Loungers, cabanas & café, lit up at night"),
                       ("Safe & supervised", "Trained lifeguard, open 6 AM to 10 PM")]:
            st.markdown(f"<div class='feature'><h4>{t_}</h4>{d_}</div><br>", unsafe_allow_html=True)
    st.subheader("🛝 Multiple parks for kids")
    k1, k2 = st.columns([1.5, 1], gap="large")
    with k1:
        st.iframe(slider_html(["park", "kids", "colony", "highway"], height=320), height=340)
    with k2:
        for t_, d_ in [("Play zones", "Swings, slides, see-saw & sandpit with soft rubber flooring"),
                       ("Toddler area", "Separate, fenced zone with shaded seating for parents"),
                       ("Active kids", "Skating & cycling track, mini sports court")]:
            st.markdown(f"<div class='feature'><h4>{t_}</h4>{d_}</div><br>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Colony": n, "Type": v["tier"], "Swimming pool": "✅" if v["pool"] else "—",
                                "Kids' parks": v["parks"]} for n, v in COLONIES.items()]).set_index("Colony"),
                 width="stretch")

# ============================================================
#  TAB: INTERIORS (hall, bedroom, kitchen, dining, furniture, bathroom)
# ============================================================
with tab_int:
    st.subheader("🛋️ Step inside: living area, bedrooms, kitchen & furniture")
    ROOMS = {
        "hall": ("🛋️ Living Hall", ["Large double-height or open hall", "Big windows with natural light", "Space for sofa set, TV unit & dining", "Vitrified / marble flooring"]),
        "bedroom": ("🛏️ Bedroom", ["Master bedroom with attached bathroom", "Wardrobe & study corner", "Balcony access in premium homes", "Quiet, cross-ventilated rooms"]),
        "kitchen": ("🍳 Kitchen", ["Modular kitchen with chimney & hob", "Granite / quartz countertop", "Utility area & storage lofts", "Island counter in luxury villas"]),
        "dining": ("🍽️ Dining", ["Open dining next to the kitchen", "4 / 6 seater dining set", "Pendant lighting & crockery unit", "Direct access to balcony or garden"]),
        "furniture": ("🪑 Furniture", ["Semi & fully furnished options", "Sofa sets, beds, wardrobes, TV unit", "Branded fittings & fans", "Choose your own interior designer"]),
        "bathroom": ("🛁 Bathroom", ["Anti-skid tiles & rain shower", "Geyser & premium sanitary-ware", "Large mirror & vanity", "Bathtub in luxury villas"]),
    }
    room_tabs = st.tabs([v[0] for v in ROOMS.values()])
    for rt, (rk, (rname, feats)) in zip(room_tabs, ROOMS.items()):
        with rt:
            r1, r2 = st.columns([1.6, 1], gap="large")
            with r1:
                st.iframe(slider_html([rk], height=360, interval=3200), height=380)
            with r2:
                st.markdown(f"<div class='feature'><h4>{rname}</h4>" + "".join(f"✔ {f}<br>" for f in feats) + "</div>",
                            unsafe_allow_html=True)
                st.caption("Furnishing level in the Price tab: Unfurnished, Semi or Fully furnished changes the estimate.")
    st.markdown("#### Complete home tour")
    st.iframe(slider_html(["hall", "bedroom", "kitchen", "dining", "furniture", "bathroom"], height=420, interval=3000), height=440)

# ============================================================
#  TAB 4: GALLERY
# ============================================================
with tab4:
    st.subheader("Photo gallery")
    st.iframe(slider_html(["home", "hall", "bedroom", "kitchen", "plot", "luxury", "luxury2", "colony", "highway", "pool", "kids", "club"], height=520, interval=3500), height=540)
    st.caption("Photos: Unsplash (free licence). Add your own project photos to the images folder to replace them.")

# ============================================================
#  TAB 5: BOOK A VISIT
# ============================================================
with tab5:
    st.subheader("📞 Book a site visit")
    with st.form("visit"):
        f1, f2 = st.columns(2)
        nm = f1.text_input("Your name")
        ph = f2.text_input("Phone number")
        f3, f4 = st.columns(2)
        col_ = f3.selectbox("Colony", list(COLONIES.keys()))
        when = f4.date_input("Preferred visit date")
        want = st.multiselect("Interested in", ["Plot", "Ready home", "Swimming pool tour", "Kids' park tour", "Loan help"])
        sent = st.form_submit_button("Request visit", type="primary")
    if sent:
        if nm and ph:
            st.success(f"Thank you {nm}! Visit request for {col_} on {when:%d %b %Y} received. We will call {ph} shortly.")
            st.caption("Demo form: connect it to email, Google Sheets or a database to store real enquiries.")
        else:
            st.error("Please enter your name and phone number.")

# ============================================================
#  TAB 6: ADD YOUR OWN COLONY + UPLOAD PHOTOS (no folders needed)
# ============================================================
def save_photos(files, scene_key):
    n = len(glob.glob(os.path.join(IMG_DIR, f"{scene_key}_*")))
    os.makedirs(IMG_DIR, exist_ok=True)
    for f in files:
        ext = f.name.rsplit(".", 1)[-1].lower()
        n += 1
        with open(os.path.join(IMG_DIR, f"{scene_key}_{n}.{ext}"), "wb") as fh:
            fh.write(f.getbuffer())
    photo_uris.clear()


def to_ints(txt, default):
    vals = [int(x) for x in re.findall(r"\d+", txt)]
    return sorted(set(vals)) or default


with tab6:
    st.subheader("📸 Upload photos for a colony")
    st.caption("Pick a colony, upload its real photos, and they appear in every slider. No folders needed.")
    u1, u2 = st.columns([1, 2])
    target = u1.selectbox("Colony", list(COLONIES.keys()), key="photo_target")
    ups = u2.file_uploader("Choose photos (JPG / PNG / WEBP)", type=["jpg", "jpeg", "png", "webp"],
                           accept_multiple_files=True, key="photo_files")
    if st.button("Save photos", type="primary") and ups:
        save_photos(ups, COLONIES[target]["scene"])
        st.success(f"{len(ups)} photo(s) saved for {target}. Open the Colonies tab to see them.")
        st.rerun()

    st.divider()
    st.subheader("🧹 Stock photos: remove any photo you don't like")
    st.caption("Untick a photo to hide it everywhere in the app, then press Save.")
    keep = {}
    for key, urls in REMOTE_PHOTOS.items():
        if not urls:
            continue
        st.markdown(f"**{CAPTIONS[key]}**")
        cols_ = st.columns(min(len(urls), 5))
        for j, url in enumerate(urls):
            with cols_[j % len(cols_)]:
                st.image(url, width="stretch")
                keep[url] = st.checkbox("Show", value=url not in HIDDEN, key=f"keep_{key}_{j}")
    if st.button("Save photo choices", type="primary"):
        with open(STOCK_FILE, "w", encoding="utf-8") as fh:
            json.dump(sorted(u for u, k_ in keep.items() if not k_), fh)
        st.success("Saved. The unticked photos are now hidden.")
        st.rerun()

    st.divider()
    st.subheader("➕ Add your own colony")
    st.caption("Example: add your real colony (name, city, sizes, amenities) and upload its photos. "
               "Enter only facts you know are correct.")
    with st.form("add_colony", clear_on_submit=False):
        a1, a2, a3 = st.columns(3)
        n_name = a1.text_input("Colony name")
        n_city = a2.text_input("City")
        n_tier = a3.selectbox("Type", ["Normal Colony", "Luxury Colony"])
        b1, b2, b3, b4 = st.columns(4)
        n_hw = b1.radio("Location", ["Inside colony", "On highway"])
        n_dc = b2.number_input("Distance from city centre (km)", 0.5, 25.0, 8.0, 0.5)
        n_dh = b3.number_input("Distance from highway (km)", 0.0, 8.0, 1.0, 0.1)
        n_rw = b4.selectbox("Road width (ft)", [15, 20, 30, 40, 60, 80, 100, 120], index=2)
        c1, c2 = st.columns(2)
        n_plots = c1.text_input("Plot sizes (sqft, comma separated)", "1000, 1500, 2000")
        n_homes = c2.text_input("Home sizes (sqft, comma separated)", "1200, 1800, 2400")
        d1, d2, d3 = st.columns(3)
        n_pool = d1.checkbox("Swimming pool")
        n_parks = d2.number_input("Kids' parks", 0, 10, 1)
        n_prem = d3.slider("Amenity premium (%)", 0, 15, 0)
        n_am = st.text_area("Amenities (one per line)", "24x7 security\nKids play park\nStreet lights")
        n_tag = st.text_input("Short tagline", "")
        n_files = st.file_uploader("Colony photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
                                   key="new_colony_files")
        add = st.form_submit_button("Add colony", type="primary")
    if add:
        if not n_name.strip():
            st.error("Please enter the colony name.")
        else:
            label = f"{n_name.strip()}, {n_city.strip()}" if n_city.strip() else n_name.strip()
            slug = "c_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            plots = to_ints(n_plots, [1000, 1500, 2000])
            homes = to_ints(n_homes, [1200, 1800, 2400])
            on_hw = 1 if n_hw == "On highway" else 0
            CUSTOM[label] = {
                "tier": n_tier, "emoji": "🏡", "color": "#d4af37", "accent": "#f3d36b",
                "tagline": n_tag or f"A {n_tier.lower()} in {n_city or 'your city'}",
                "on_hw": on_hw, "d_hw": 0.1 if on_hw else float(max(n_dh, 0.5)), "road_w": int(n_rw),
                "d_city": float(n_dc), "premium": n_prem / 100,
                "plots": plots, "homes": homes,
                "amenities": [x.strip() for x in n_am.splitlines() if x.strip()] or ["Gated colony"],
                "pool": bool(n_pool), "parks": int(n_parks), "scene": slug,
            }
            save_custom(CUSTOM)
            if n_files:
                save_photos(n_files, slug)
            st.success(f"✅ {label} added. Check the Colonies and Price & EMI tabs.")
            st.rerun()

    if CUSTOM:
        st.divider()
        dl1, dl2 = st.columns([2, 1])
        gone = dl1.selectbox("Remove one of your colonies", list(CUSTOM.keys()))
        if dl2.button("Remove colony"):
            CUSTOM.pop(gone, None)
            save_custom(CUSTOM)
            st.rerun()
    st.caption("Note: on free cloud hosting, saved colonies and photos can reset when the app restarts. "
               "For permanent storage, keep the images folder and custom_colonies.json in your project.")

st.markdown("<div class='footer'>Royal Estates · demo project · price estimates by Decision Tree model (R² ≈ 0.91)</div>",
            unsafe_allow_html=True)
