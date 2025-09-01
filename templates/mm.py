# app.py
from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import csv
import io
import base64
import joblib
import bcrypt
import pandas as pd
import numpy as np

# Matplotlib (server-friendly)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from werkzeug.utils import secure_filename

# ML imports (used in admin upload retraining)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Words conversion
from num2words import num2words

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'secret123'  # Change in production!

# Uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'csv', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Data files
CSV_FILE = "predictions.csv"
MODEL_PATH = "models/best_house_price_model.pkl"
DATA_PATH = "House_Price_Data.csv"

# Create predictions CSV if missing/empty
if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
    pd.DataFrame(columns=[
        "ID", "City", "BHK", "Property Type", "SqFt",
        "Predicted Price (in ₹)", "Predicted Price (in words)", "User"
    ]).to_csv(CSV_FILE, index=False)

# Cities list
CITIES = [
    'Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Ahmedabad', 'Sembakkam','Chennai', 'Theni',
    'Anna Nagar','Ambattur','East Tambaram','Vadapalani','Porur','Kolkata', 'Pune', 'Jaipur',
    'Surat', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore','Thane', 'Bhopal', 'Visakhapatnam',
    'Vadodara', 'Ghaziabad', 'Ludhiana', 'Agra','Nashik', 'Faridabad', 'Meerut', 'Rajkot',
    'Kalyan', 'Vasai-Virar', 'Varanasi','Srinagar', 'Aurangabad', 'Dhanbad', 'Amritsar',
    'Navi Mumbai', 'Allahabad','Ranchi', 'Howrah', 'Coimbatore', 'Jabalpur', 'Gwalior',
    'Vijayawada', 'Jodhpur','Madurai', 'Raipur', 'Kota', 'Guwahati', 'Chandigarh', 'Solapur',
    'Hubli–Dharwad','Bareilly', 'Mysore'
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Set to 'lakhs', 'rupees', or 'auto'
PREDICTION_UNIT = 'auto'

def convert_prediction_to_rupees(pred_value: float) -> int:
    """
    Convert model prediction to rupees using a robust rule:
    - If PREDICTION_UNIT == 'lakhs': multiply by 1e5
    - If PREDICTION_UNIT == 'rupees': return as is
    - If 'auto':
        Assume lakhs if prediction looks like a typical lakhs value (< 2,000).
        Otherwise assume rupees.
    """
    if PREDICTION_UNIT == 'lakhs':
        return int(round(pred_value * 100000))
    elif PREDICTION_UNIT == 'rupees':
        return int(round(pred_value))
    else:
        # auto
        # Heuristic: typical lakhs values are usually < 2,000 (i.e., < ₹200 cr)
        if pred_value < 2000:
            return int(round(pred_value * 100000))
        else:
            return int(round(pred_value))

def format_indian(n: int) -> str:
    """
    Format integer in Indian numbering style (e.g., 12,34,567).
    """
    s = str(n)
    if len(s) <= 3:
        return s
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    return ','.join(parts[::-1]) + ',' + last3

def to_words_in_indian_rupees(n: int) -> str:
    try:
        # num2words en_IN handles lakh/crore wording
        words = num2words(n, lang='en_IN')
        # ensure clean spacing/commas
        return f"{words} rupees".replace(",", "")
    except Exception as e:
        return f"Unavailable ({e})"

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    # If model missing, create a minimal placeholder model to avoid crashes
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # Build a tiny pipeline to be replaced later via /admin_upload
    dummy_pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['location', 'propertytype']),
            ('num', 'passthrough', ['bhk', 'sqft'])
        ])),
        ('model', LGBMRegressor())
    ])
    # Train dummy if data exists
    if os.path.exists(DATA_PATH):
        try:
            df_all = pd.read_csv(DATA_PATH)
            # Try to infer columns
            for c in ['location','bhk','propertytype','sqft','price']:
                if c not in df_all.columns:
                    raise ValueError("Training data missing required columns.")
            clean = df_all[['location','bhk','propertytype','sqft','price']].dropna()
            if not clean.empty:
                X = clean[['location','bhk','propertytype','sqft']]
                y = clean['price']
                dummy_pipeline.fit(X, y)
                joblib.dump(dummy_pipeline, MODEL_PATH)
        except Exception:
            pass

# Load trained model (assumes saved pipeline)
model = joblib.load(MODEL_PATH)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

# Keep '/' as home; login lives at '/login'
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/user_home')
def user_home():
    if 'user' in session:
        if session.get("user") == "admin":
            return redirect(url_for('admin'))
        else:
            return redirect(url_for('index'))
    return redirect(url_for('login'))

# Login / Register
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None

    if request.method == 'POST':
        uname = request.form.get('username', '').strip()
        pwd = request.form.get('password', '')
        is_register = request.form.get('register')

        # Admin
        if uname == 'admin' and pwd == '1234':
            session['user'] = 'admin'
            return redirect(url_for('admin'))

        # Ensure users.csv exists
        if not os.path.exists('users.csv'):
            pd.DataFrame(columns=['username', 'password']).to_csv('users.csv', index=False)

        users_df = pd.read_csv('users.csv')

        if is_register:
            # Register flow
            if uname == '' or pwd == '':
                error = "Username and password are required."
            elif uname in users_df['username'].values:
                error = "Username already exists. Choose a different one."
            else:
                hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
                pd.DataFrame([[uname, hashed]], columns=['username', 'password']) \
                    .to_csv('users.csv', mode='a', header=False, index=False)
                session['user'] = uname
                return redirect(url_for('index'))
        else:
            # Login flow
            if uname in users_df['username'].values:
                stored_hash = users_df.loc[users_df['username'] == uname, 'password'].values[0]
                if bcrypt.checkpw(pwd.encode(), stored_hash.encode()):
                    session['user'] = uname
                    return redirect(url_for('user_home'))
                else:
                    error = "Incorrect password."
            else:
                error = "Username not found. Please register."

    return render_template('login.html', error=error)

# Prediction page
@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user' not in session or session.get('user') == 'admin':
        return redirect(url_for('login'))

    prediction = prediction_words = None
    house_images = []

    if request.method == 'POST':
        city = request.form.get('location', '').strip()
        bhk = request.form.get('BHK', '').strip()
        prop = request.form.get('propertytype', '').strip()
        sqft = request.form.get('sqft', '').strip()

        # Basic validation
        try:
            bhk = int(float(bhk))
            sqft = float(sqft)
        except Exception:
            flash("Please enter valid numeric values for BHK and SqFt.", 'danger')
            return render_template("index.html", cities=CITIES,
                                   prediction=None, prediction_words=None,
                                   house_images=[])

        input_df = pd.DataFrame([{
            'location': city,
            'bhk': bhk,
            'propertytype': prop,
            'sqft': sqft
        }])

        try:
            raw_pred = float(model.predict(input_df)[0])
            price_in_rupees = convert_prediction_to_rupees(raw_pred)

            # Guardrails: clip absurd values
            # (e.g., between ₹1 lakh and ₹500 crore)
            price_in_rupees = int(np.clip(price_in_rupees, 1e5, 5e9))

            prediction = f"₹{format_indian(price_in_rupees)}"
            prediction_words = to_words_in_indian_rupees(price_in_rupees)

            # Save to CSV
            if os.path.exists(CSV_FILE):
                df_existing = pd.read_csv(CSV_FILE)
            else:
                df_existing = pd.DataFrame(columns=[
                    "ID", "City", "BHK", "Property Type", "SqFt",
                    "Predicted Price (in ₹)", "Predicted Price (in words)", "User"
                ])

            new_id = (df_existing['ID'].max() + 1) if not df_existing.empty else 1
            new_row = {
                "ID": int(new_id),
                "City": city,
                "BHK": bhk,
                "Property Type": prop,
                "SqFt": sqft,
                "Predicted Price (in ₹)": price_in_rupees,
                "Predicted Price (in words)": prediction_words,
                "User": session.get("user", "user")
            }

            df_updated = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
            df_updated.to_csv(CSV_FILE, index=False)

        except Exception as e:
            prediction = "Error in prediction"
            prediction_words = f"Unavailable ({e})"

        # Load relevant images
        folder_name = f"{city}_{bhk}bhk_{prop}".replace(" ", "").lower()
        custom_folder = os.path.join('static', 'house_images', folder_name)

        if os.path.exists(custom_folder) and any(
            f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(custom_folder)
        ):
            house_images = [
                url_for('static', filename=f"house_images/{folder_name}/{img}")
                for img in os.listdir(custom_folder)
                if img.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
        else:
            default_folder = os.path.join('static', 'house_images', 'default')
            if os.path.exists(default_folder):
                house_images = [
                    url_for('static', filename=f"house_images/default/{img}")
                    for img in os.listdir(default_folder)
                    if img.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
            else:
                house_images = []

    return render_template("index.html", cities=CITIES,
                           prediction=prediction,
                           prediction_words=prediction_words,
                           house_images=house_images)

# User history
@app.route('/user_history')
def user_history():
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']
    data = []

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('User') == username:
                    try:
                        price = float(row['Predicted Price (in ₹)'])
                        row['Predicted Price (in ₹)'] = f"₹{format_indian(int(price))}"
                    except (ValueError, TypeError):
                        pass
                    data.append(row)

    if not data:
        return render_template("user_history.html", message="No records found.")

    return render_template("user_history.html", records=data)

# Admin dashboard
@app.route('/admin')
def admin():
    if session.get("user") != "admin":
        return redirect(url_for('login'))

    try:
        df = pd.read_csv(CSV_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=[
            "ID", "City", "BHK", "Property Type", "SqFt",
            "Predicted Price (in ₹)", "Predicted Price (in words)", "User"
        ])
        df.to_csv(CSV_FILE, index=False)

    # Format prices for display
    if not df.empty and 'Predicted Price (in ₹)' in df.columns:
        df['Predicted Price (in ₹)'] = pd.to_numeric(df['Predicted Price (in ₹)'], errors='coerce')
        df['Predicted Price (in ₹)'] = df['Predicted Price (in ₹)'].apply(
            lambda x: f"₹{format_indian(int(x))}" if pd.notnull(x) else ''
        )

    return render_template("admin.html", records=df.to_dict(orient='records'))

# Upload + retrain
@app.route('/admin_upload', methods=['GET', 'POST'])
def upload():
    if session.get("user") != "admin":
        return redirect(url_for('login'))

    table_data = None
    table_columns = None
    uploaded_preview = None
    uploaded_columns = None

    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if not file or not file.filename:
                flash('Please select a file.', 'danger')
                return redirect(url_for('upload'))

            if not file.filename.lower().endswith('.csv'):
                flash('Only CSV files are allowed', 'danger')
                return redirect(url_for('upload'))

            # Read CSV and normalize col names
            new_data = pd.read_csv(file)
            new_cols_map = {c: c.strip().lower() for c in new_data.columns}
            new_data.rename(columns=new_cols_map, inplace=True)

            # Required columns for training
            expected = {'location', 'bhk', 'propertytype', 'sqft', 'price'}
            if not expected.issubset(set(new_data.columns)):
                miss = expected.difference(set(new_data.columns))
                flash(f"Missing columns: {', '.join(sorted(miss))}", 'danger')
                return redirect(url_for('upload'))

            new_data = new_data.dropna(subset=['location','bhk','propertytype','sqft','price'])

            # Combine with existing data (if available)
            if os.path.exists(DATA_PATH):
                try:
                    old_data = pd.read_csv(DATA_PATH)
                    combined = pd.concat([old_data, new_data], ignore_index=True)
                except Exception:
                    combined = new_data.copy()
            else:
                combined = new_data.copy()

            combined.to_csv(DATA_PATH, index=False)

            # Preview for UI
            uploaded_preview = new_data.head(20).to_dict(orient='records')
            uploaded_columns = list(new_data.columns)

            # Retrain model
            clean = combined[['location','bhk','propertytype','sqft','price']].dropna()
            X = clean[['location','bhk','propertytype','sqft']]
            y = clean['price']

            pipeline = Pipeline([
                ('preprocessor', ColumnTransformer(transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), ['location', 'propertytype']),
                    ('num', 'passthrough', ['bhk', 'sqft'])
                ])),
                ('model', LGBMRegressor(random_state=42))
            ])

            pipeline.fit(X, y)

            # Save updated model
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(pipeline, MODEL_PATH)
            # Reload global model
            global model
            model = joblib.load(MODEL_PATH)

            flash("Model updated and saved successfully!", 'success')

        except Exception as e:
            flash(f"Error processing CSV: {e}", 'danger')

    # Show combined table (first 20 rows)
    try:
        df = pd.read_csv(DATA_PATH)
        table_columns = df.columns.tolist()
        table_data = df.head(20).to_dict(orient='records')
    except Exception:
        if not uploaded_preview:
            flash("No data available to display.", 'warning')
        table_data = None
        table_columns = None

    return render_template('upload.html',
                           table_data=table_data,
                           table_columns=table_columns,
                           uploaded_data=uploaded_preview,
                           uploaded_columns=uploaded_columns)

# Compare two predictions (admin)
@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if session.get("user") != "admin":
        return redirect(url_for('login'))

    try:
        df = pd.read_csv(CSV_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=[
            "ID", "City", "BHK", "Property Type", "SqFt",
            "Predicted Price (in ₹)", "Predicted Price (in words)", "User"
        ])
        df.to_csv(CSV_FILE, index=False)

    ids = df['ID'].dropna().astype(int).tolist() if not df.empty and 'ID' in df.columns else []
    row1 = row2 = None

    if request.method == "POST":
        try:
            id1 = int(request.form.get('id1', 0))
            id2 = int(request.form.get('id2', 0))
            if id1 in df['ID'].values and id2 in df['ID'].values:
                row1 = df[df['ID'] == id1].iloc[0].to_dict()
                row2 = df[df['ID'] == id2].iloc[0].to_dict()
        except Exception:
            pass

    return render_template("compare.html", ids=ids, row1=row1, row2=row2)

# Admin graph (avg price by city)
@app.route('/graph')
def graph():
    if session.get("user") != "admin":
        return redirect(url_for('login'))

    if not os.path.exists(CSV_FILE):
        return "No data available"

    df = pd.read_csv(CSV_FILE)
    if df.empty or 'City' not in df.columns or 'Predicted Price (in ₹)' not in df.columns:
        return "No data available"

    df['price_clean'] = pd.to_numeric(df['Predicted Price (in ₹)'], errors='coerce')
    avg_prices = df.groupby('City')['price_clean'].mean().dropna().sort_values(ascending=False)
    if avg_prices.empty:
        return "No data available"

    return render_template("graph.html",
                           labels=avg_prices.index.tolist(),
                           values=[int(x) for x in avg_prices.values.tolist()])

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/whoami')
def whoami():
    return f"Logged in as: {session.get('user')}"

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # In production, use a WSGI server instead (gunicorn/uwsgi)
    app.run(debug=True)
