# /workspace/app/app.py
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from dash import Dash, dcc, html, Input, Output, State

# ---------- paths ----------
APP_DIR    = Path(__file__).parent
MODEL_PATH = APP_DIR / "RandomForestRegressor.z"
META_PATH  = APP_DIR / "model_meta.json"  # optional

# ---------- load model ----------
model = load(MODEL_PATH)  # Pipeline wrapped with TransformedTargetRegressor

# columns expected (pre-OHE)
CAT_COLS = ["fuel", "seller_type", "transmission", "brand"]
NUM_COLS = ["year", "km_driven", "owner", "engine", "max_power", "mileage", "seats"]
ALL_COLS = CAT_COLS + NUM_COLS

# ---------- metadata / options ----------
meta: dict = {}
if META_PATH.exists():
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

def opt(key: str, fallback: list[str]):
    vals = meta.get("unique_values", {}).get(key)
    return sorted(vals) if isinstance(vals, list) and vals else fallback

# จำกัด fuel ให้ตรงกับที่ train จริง
fuel_opts         = opt("fuel", ["Diesel", "Petrol"])
seller_type_opts  = opt("seller_type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission_opts = opt("transmission", ["Manual", "Automatic"])
brand_opts        = opt("brand", [
    'Maruti','Skoda','Honda','Hyundai','Toyota','Ford','Renault','Mahindra','Tata',
    'Chevrolet','Fiat','Datsun','Jeep','Mercedes-Benz','Mitsubishi','Audi','Volkswagen',
    'BMW','Nissan','Lexus','Jaguar','Land','MG','Volvo','Daewoo','Kia','Force',
    'Ambassador','Ashok','Isuzu','Opel','Peugeot'
])

# mapping Owner (ให้ผู้ใช้เลือกเป็นคำ แต่ส่งเลขให้โมเดล)
OWNER_MAP = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 5,
}
OWNER_OPTIONS = list(OWNER_MAP.keys())

# ค่าตั้งต้น
defaults = {
    "fuel": "Diesel",
    "seller_type": "Dealer",
    "transmission": "Manual",
    "brand": brand_opts[0] if brand_opts else "Toyota",
    "year": 2018,
    "km_driven": 40000,
    "owner_txt": "First Owner",   # default สำหรับ dropdown owner แบบคำ
    "engine": 1496,
    "max_power": 110,
    "mileage": 19.5,
    "seats": 5,
} | meta.get("defaults", {})

OWNER_NOTE = (
    "Owner mapping: 1=First Owner, 2=Second Owner, 3=Third Owner, "
    "4=Fourth & Above Owner, 5=Test Drive Car"
)

# ---------- UI helpers ----------
def num_input(id_, value, *, step=1, min_=0, max_=None, label="", note=None):
    return html.Div([
        html.Label(label),
        dcc.Input(
            id=id_, type="number", value=value, step=step, min=min_, max=max_,
            style={"width": "100%"}
        ),
        html.Small(note, style={"color": "#555"}) if note else None,
    ], style={"marginBottom": "10px"})

def dd(id_, options, value, label=""):
    # ถ้า options ว่างให้ใช้ input text
    if options:
        return html.Div([
            html.Label(label),
            dcc.Dropdown(
                id=id_,
                options=[{"label": v, "value": v} for v in options],
                value=value, searchable=True, clearable=False
            )
        ], style={"marginBottom": "10px"})
    else:
        return html.Div([
            html.Label(label),
            dcc.Input(id=id_, type="text", value=value, style={"width": "100%"})
        ], style={"marginBottom": "10px"})

# ---------- Dash app ----------
app = Dash(__name__)
server = app.server  # for production servers

app.layout = html.Div(
    style={"maxWidth": "900px", "margin": "30px auto", "fontFamily": "system-ui, sans-serif"},
    children=[
        html.H2("Car Price Predictor"),
        html.P("Enter the vehicle information and click Predict to estimate the selling price (powered by a RandomForest model)"),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                dd("fuel",         fuel_opts,         defaults["fuel"],         "Fuel"),
                dd("seller_type",  seller_type_opts,  defaults["seller_type"],  "Seller Type"),
                dd("transmission", transmission_opts, defaults["transmission"], "Transmission"),
                dd("brand",        brand_opts,        defaults["brand"],        "Brand"),
                dd("owner_txt", OWNER_OPTIONS, defaults.get("owner_txt", "First Owner"), "Owner"),
                num_input("year",       defaults["year"],       step=1,   min_=1990,  label="Year"),
                num_input("km_driven",  defaults["km_driven"],  step=1, min_=0,     label="KM Driven"),
                num_input("engine",     defaults["engine"],     step=10,  min_=500,   label="Engine (CC)"),
                num_input("max_power",  defaults["max_power"],  step=1,   min_=20,    label="Max Power (hp)"),
                num_input("mileage",    defaults["mileage"],    step=0.1, min_=0,     label="Mileage (kmpl)"),
                num_input("seats",      defaults["seats"],      step=1,   min_=2,     label="Seats"),
            ]
        ),

        html.Button("Predict", id="btn", n_clicks=0, style={"marginTop": "12px"}),
        html.H3(id="out", style={"marginTop": "16px", "color": "#0b6"}),
        html.Pre(id="debug", style={
            "background": "#f7f7f7", "padding": "12px", "border": "1px solid #eee", "whiteSpace": "pre-wrap"
        }),
    ]
)

@app.callback(
    Output("out", "children"),
    Output("debug", "children"),
    Input("btn", "n_clicks"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("brand", "value"),
    State("year", "value"),
    State("km_driven", "value"),
    State("owner_txt", "value"),     # รับ owner เป็นคำ
    State("engine", "value"),
    State("max_power", "value"),
    State("mileage", "value"),
    State("seats", "value"),
    prevent_initial_call=True
)
def do_predict(n, fuel, seller_type, transmission, brand,
               year, km_driven, owner_txt, engine, max_power, mileage, seats):

    # map owner (คำ) -> เลขสำหรับโมเดล
    owner_num = OWNER_MAP.get(owner_txt, 1)

    payload = {
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "brand": brand,
        "year": year,
        "km_driven": km_driven,
        "owner": owner_num,  # ส่งเลขให้โมเดล
        "engine": engine,
        "max_power": max_power,
        "mileage": mileage,
        "seats": seats,
    }

    # สร้าง DataFrame 1 แถวตาม ALL_COLS
    row = {c: None for c in ALL_COLS}
    row.update(payload)
    X = pd.DataFrame([row], columns=ALL_COLS)

    # แปลง numeric ให้ชัวร์

    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # guard: check if any field is NaN → stop early and tell user
    missing = [c for c in ALL_COLS if X[c].isna().any()]
    if missing:
        return ("⚠️ Please correct the highlighted fields.",
                f"Missing/invalid: {', '.join(missing)}")

    # ปรับบางคอลัมน์กันผิดพลาด
    X["owner"] = X["owner"].clip(1, 5)
    X["year"] = X["year"].clip(1990)

    try:
        pred = float(model.predict(X)[0])  # model มี preprocess + OHE + imputer + log-transform
        msg = f"Predicted price: {pred:,.0f}"
        dbg = json.dumps({**payload, "owner_text": owner_txt}, indent=2, ensure_ascii=False)
        return msg, dbg
    except Exception as e:
        return "⚠️ Prediction failed.", f"{type(e).__name__}: {e}"

if __name__ == "__main__":
    # Dash v3
    app.run(host="0.0.0.0", port=8050, debug=True)