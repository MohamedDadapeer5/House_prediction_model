from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
import pickle
import os
import re
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Global variables for the models
models = {}
scaler = None
label_encoders = {}
data = None

# Simple user database (in production, use a real database)
users_db = {}

def load_model_and_data():
    """Load the trained models and data"""
    global models, scaler, label_encoders, data
    
    # Load the dataset
    data = pd.read_csv('Bengaluru_House_Data.csv')
    
    # Preprocess data (same as in main script)
    df = data.copy()
    
    # Handle missing values - Fix pandas warnings
    df.loc[:, 'society'] = df['society'].fillna('Unknown')
    df.loc[:, 'balcony'] = df['balcony'].fillna(df['balcony'].median())
    df.loc[:, 'bath'] = df['bath'].fillna(df['bath'].median())
    
    # Clean total_sqft column - Improved function
    def clean_sqft(x):
        if pd.isna(x):
            return np.nan
        
        if isinstance(x, str):
            # Remove any text like "Sq. Meter", "sqft", etc.
            x = re.sub(r'[^\d.\-]', '', x)
            
            if '-' in x:
                # Take average of range
                parts = x.split('-')
                try:
                    return (float(parts[0].strip()) + float(parts[1].strip())) / 2
                except:
                    return np.nan
            else:
                try:
                    return float(x)
                except:
                    return np.nan
        elif isinstance(x, (int, float)):
            return float(x)
        else:
            return np.nan
    
    df['total_sqft'] = df['total_sqft'].apply(clean_sqft)
    
    # Remove rows with invalid total_sqft
    df = df.dropna(subset=['total_sqft'])
    
    # Extract bedrooms
    def extract_bedrooms(size):
        if pd.isna(size):
            return 2  # Default value
        
        if isinstance(size, str):
            if 'BHK' in size:
                try:
                    return int(size.split()[0])
                except:
                    return 2
            elif 'Bedroom' in size:
                try:
                    return int(size.split()[0])
                except:
                    return 2
            elif 'RK' in size:
                return 1
        return 2  # Default value
    
    df['bedrooms'] = df['size'].apply(extract_bedrooms)
    
    # Handle availability
    df['is_ready_to_move'] = (df['availability'] == 'Ready To Move').astype(int)
    
    # Create location features
    location_counts = df['location'].value_counts()
    df['location_frequency'] = df['location'].map(location_counts)
    
    # Encode categorical variables
    categorical_cols = ['area_type', 'location']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Select features
    feature_cols = [
        'total_sqft', 'bath', 'balcony', 'bedrooms', 'is_ready_to_move',
        'location_frequency', 'area_type_encoded', 'location_encoded'
    ]
    
    # Remove rows with missing values
    df_clean = df[feature_cols + ['price']].dropna()
    
    # Prepare training data
    X = df_clean[feature_cols]
    y = df_clean['price']

    # Create scaler (for consistency with main script)
    scaler = StandardScaler()
    scaler.fit(X)

    # Train all models
    models.clear()

    # Linear Regression (use scaled features)
    lr = LinearRegression()
    lr.fit(scaler.transform(X), y)
    models['Linear Regression'] = lr
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    models['Random Forest'] = rf
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X, y)
    models['Gradient Boosting'] = gb
    
    # XGBoost
    if XGBRegressor is not None:
        xgb = XGBRegressor(n_estimators=100, random_state=42)
        xgb.fit(X, y)
        models['XGBoost'] = xgb
    
    print("All models loaded successfully!")
    print(f"Training data shape: {X.shape}")
    print(f"Features used: {feature_cols}")

def get_sample_properties():
    """Get sample properties for display on home page - Exactly 9 cards"""
    global data
    if data is None:
        return []
    
    # Initialize with exactly 9 slots
    sample_properties = []
    target_count = 9
    
    # Try to get specific locations first
    try:
        # Vijayanagar properties (up to 2)
        vijayanagar_props = data[data['location'].str.contains('Vijayanagar', case=False, na=False)]
        if len(vijayanagar_props) > 0:
            vijayanagar_sample = vijayanagar_props.sample(min(2, len(vijayanagar_props)))
            for _, row in vijayanagar_sample.iterrows():
                if len(sample_properties) < target_count:
                    sample_properties.append({
                        'location': row['location'],
                        'area_type': row['area_type'],
                        'size': row['size'],
                        'total_sqft': row['total_sqft'],
                        'price': row['price'],
                        'bath': row['bath'],
                        'balcony': row['balcony'],
                        'category': 'Premium' if row['price'] > 100 else 'Mid-Range' if row['price'] >= 50 else 'Budget'
                    })
        
        # Dodda Nekkundi properties (up to 2)
        dodda_nekkundi_props = data[data['location'].str.contains('Dodda Nekkundi', case=False, na=False)]
        if len(dodda_nekkundi_props) > 0:
            dodda_nekkundi_sample = dodda_nekkundi_props.sample(min(2, len(dodda_nekkundi_props)))
            for _, row in dodda_nekkundi_sample.iterrows():
                if len(sample_properties) < target_count:
                    sample_properties.append({
                        'location': row['location'],
                        'area_type': row['area_type'],
                        'size': row['size'],
                        'total_sqft': row['total_sqft'],
                        'price': row['price'],
                        'bath': row['bath'],
                        'balcony': row['balcony'],
                        'category': 'Premium' if row['price'] > 100 else 'Mid-Range' if row['price'] >= 50 else 'Budget'
                    })
    except:
        pass
    
    # Fill remaining slots to get exactly 9 properties
    remaining_slots = target_count - len(sample_properties)
    
    if remaining_slots > 0:
        # Calculate how many of each category to add
        premium_count = max(1, remaining_slots // 3)
        mid_range_count = max(1, remaining_slots // 3)
        budget_count = remaining_slots - premium_count - mid_range_count
        
        # Add Premium properties
        premium_data = data[data['price'] > 100]
        if len(premium_data) > 0:
            premium_sample = premium_data.sample(min(premium_count, len(premium_data)))
            for _, row in premium_sample.iterrows():
                if len(sample_properties) < target_count:
                    sample_properties.append({
                        'location': row['location'],
                        'area_type': row['area_type'],
                        'size': row['size'],
                        'total_sqft': row['total_sqft'],
                        'price': row['price'],
                        'bath': row['bath'],
                        'balcony': row['balcony'],
                        'category': 'Premium'
                    })
        
        # Add Mid-range properties
        mid_range_data = data[(data['price'] >= 50) & (data['price'] <= 100)]
        if len(mid_range_data) > 0:
            mid_range_sample = mid_range_data.sample(min(mid_range_count, len(mid_range_data)))
            for _, row in mid_range_sample.iterrows():
                if len(sample_properties) < target_count:
                    sample_properties.append({
                        'location': row['location'],
                        'area_type': row['area_type'],
                        'size': row['size'],
                        'total_sqft': row['total_sqft'],
                        'price': row['price'],
                        'bath': row['bath'],
                        'balcony': row['balcony'],
                        'category': 'Mid-Range'
                    })
        
        # Add Budget properties
        budget_data = data[data['price'] < 50]
        if len(budget_data) > 0:
            budget_sample = budget_data.sample(min(budget_count, len(budget_data)))
            for _, row in budget_sample.iterrows():
                if len(sample_properties) < target_count:
                    sample_properties.append({
                        'location': row['location'],
                        'area_type': row['area_type'],
                        'size': row['size'],
                        'total_sqft': row['total_sqft'],
                        'price': row['price'],
                        'bath': row['bath'],
                        'balcony': row['balcony'],
                        'category': 'Budget'
                    })
    
    # If we still don't have 9 properties, fill with any available data
    while len(sample_properties) < target_count:
        random_prop = data.sample(1).iloc[0]
        sample_properties.append({
            'location': random_prop['location'],
            'area_type': random_prop['area_type'],
            'size': random_prop['size'],
            'total_sqft': random_prop['total_sqft'],
            'price': random_prop['price'],
            'bath': random_prop['bath'],
            'balcony': random_prop['balcony'],
            'category': 'Premium' if random_prop['price'] > 100 else 'Mid-Range' if random_prop['price'] >= 50 else 'Budget'
        })
    
    # Ensure exactly 9 properties and return
    return sample_properties[:target_count]

@app.route('/')
def index():
    """Landing page with animated title"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if email in users_db and users_db[email]['password'] == password:
            session['user'] = email
            session['username'] = users_db[email]['name']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password!', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page"""
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
        elif email in users_db:
            flash('Email already registered!', 'error')
        else:
            users_db[email] = {
                'name': name,
                'password': password,
                'created_at': datetime.now().isoformat()
            }
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('user', None)
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/home')
def home():
    """Home page with sample properties"""
    if 'user' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    sample_properties = get_sample_properties()
    return render_template('home.html', 
                         username=session['username'],
                         properties=sample_properties)

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """Prediction page"""
    if 'user' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        # Get unique values for dropdowns - Fix sorting issue with mixed data types
        area_types = sorted(data['area_type'].unique().tolist())
        
        # Handle location sorting with mixed data types
        locations = data['location'].unique().tolist()
        # Filter out any non-string values and sort
        locations = sorted([str(loc) for loc in locations if pd.notna(loc) and str(loc).strip()])
        
        return render_template('predict.html', 
                             username=session['username'],
                             area_types=area_types, 
                             locations=locations,
                             prediction_stats=None)
    
    # Handle POST request for prediction
    try:
        # Get form data
        area_type = request.form['area_type']
        location = request.form['location']
        size = request.form['size']
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        balcony = int(request.form['balcony'])
        availability = request.form['availability']
        
        # Preprocess the input
        def extract_bedrooms(size):
            if 'BHK' in size:
                return int(size.split()[0])
            elif 'Bedroom' in size:
                return int(size.split()[0])
            elif 'RK' in size:
                return 1
            return 2
        
        bedrooms = extract_bedrooms(size)
        is_ready_to_move = 1 if availability == 'Ready To Move' else 0
        
        # Encode categorical variables
        area_type_encoded = label_encoders['area_type'].transform([area_type])[0]
        location_encoded = label_encoders['location'].transform([location])[0]
        
        # Get location frequency
        location_counts = data['location'].value_counts()
        location_frequency = location_counts.get(location, 1)
        
        # Create feature vector
        features = np.array([[
            total_sqft, bath, balcony, bedrooms, is_ready_to_move,
            location_frequency, area_type_encoded, location_encoded
        ]])
        
        # Predict with all models
        results = {}
        for name, mdl in models.items():
            if name == 'Linear Regression':
                pred = mdl.predict(scaler.transform(features))[0]
            else:
                pred = mdl.predict(features)[0]
            results[name] = round(pred, 2)
        
        # Calculate mean of all predictions
        mean_prediction = np.mean(list(results.values()))
        
        # Find model with prediction closest to the mean (most stable/representative)
        best_model = min(results, key=lambda k: abs(results[k] - mean_prediction))
        
        # Calculate prediction statistics for better insights
        prediction_stats = {
            'mean': round(mean_prediction, 2),
            'min': round(min(results.values()), 2),
            'max': round(max(results.values()), 2),
            'range': round(max(results.values()) - min(results.values()), 2)
        }
        
        return render_template('predict.html', 
                             username=session['username'],
                             area_types=sorted(data['area_type'].unique().tolist()),
                             locations=sorted([str(loc) for loc in data['location'].unique().tolist() if pd.notna(loc) and str(loc).strip()]),
                             prediction_results=results,
                             best_model=best_model,
                             prediction_stats=prediction_stats)
        
    except Exception as e:
        return render_template('predict.html', 
                             username=session['username'],
                             area_types=sorted(data['area_type'].unique().tolist()),
                             locations=sorted([str(loc) for loc in data['location'].unique().tolist() if pd.notna(loc) and str(loc).strip()]),
                             error=str(e),
                             prediction_stats=None)

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

if __name__ == '__main__':
    # Load model when starting the app
    load_model_and_data()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 