Royal Estates: House & Plot Price Prediction (Decision Tree + Streamlit)
live app: https://house-price-predict-npxwtqncvjsuffarf48mqt.streamlit.app/
A machine learning project that predicts house and plot prices for highway and inside-colony locations, with a Streamlit web app that shows colonies, photos, swimming pool, kids' parks, interiors, an EMI calculator and a price estimator.

📁 Project files
house_project/
├── app.py                    # Streamlit web app (run this)
├── build_model.py            # Trains the model and saves the .pkl files
├── generate_dataset.py       # Creates house_price_dataset.csv (synthetic data)
├── train_decision_tree.py    # Simple training + accuracy check script
├── house_price_dataset.csv   # Dataset (1,500 rows, 12 inputs + price)
├── house_model.pkl           # Trained Decision Tree model
├── scaler.pkl                # StandardScaler used before prediction
├── columns.pkl               # Column order used by the model
├── requirements.txt          # Python packages
├── .streamlit/
│   └── config.toml           # Dark gold theme
└── images/                   # (optional) your own photos, see below

Files created by the app while you use it: custom_colonies.json (colonies you add) and stock_hidden.json (stock photos you hide).

🚀 How to run
bash
# 1. Install packages
pip install -r requirements.txt

# 2. (Only if the .pkl files are missing) build the model
python build_model.py

# 3. Start the app
streamlit run app.py

The app opens at http://localhost:8501. It needs Streamlit 1.60 or newer (it uses st.iframe).

🧠 The model
Item	Value
Algorithm	DecisionTreeRegressor
Settings	max_depth=8, min_samples_leaf=5, random_state=42
Preprocessing	StandardScaler
Split	80% train / 20% test
Accuracy (test)	R² ≈ 0.91, mean error ≈ 7.9 lakh

Input columns (12): area_sqft, bedrooms, bathrooms, balconies, floors, age_years, parking_slots, furnishing (0 = unfurnished, 1 = semi, 2 = fully), distance_city_center_km, on_highway (0/1), distance_highway_km, road_width_ft. Target: price_lakh (price in lakh INR).

What max_depth and min_samples_leaf do
max_depth: how many questions the tree may ask in a row. Too small = misses patterns, too big = memorizes the data.
min_samples_leaf: the minimum number of houses in each final group. A bigger value gives smoother, safer predictions.
🖥️ What the app has
Tab	What you can do
🏘️ Colonies	Browse colonies (Normal and Luxury) with photos, sizes, pool, kids' parks and amenities. Filter from the sidebar.
💰 Price & EMI	Choose a colony, a ready home or a plot, and the size. Get an animated price estimate and a monthly EMI. Compare all colonies.
🏊 Pool & Kids' Parks	Swimming pool and kids' park photos and features.
🛋️ Interiors	Living hall, bedroom, kitchen, dining, furniture and bathroom photos.
🖼️ Gallery	Full-width photo slider.
📞 Book a Visit	Demo enquiry form (does not store anything yet).
➕ Add Colony / Photos	Add your own colony, upload photos, and hide stock photos you don't want.
