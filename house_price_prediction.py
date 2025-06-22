import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import re
warnings.filterwarnings('ignore')

# Import XGBoost from the correct module
try:
    from xgboost import XGBRegressor
except ImportError:
    print("XGBoost not available, skipping XGBoost model")
    XGBRegressor = None

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HousePricePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.models = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("Loading Bengaluru House Data...")
        self.data = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.data.shape}")
        print("\nFirst few rows:")
        print(self.data.head())
        print("\nDataset info:")
        print(self.data.info())
        print("\nMissing values:")
        print(self.data.isnull().sum())
        print("\nBasic statistics:")
        print(self.data.describe())
        
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Create subplots for different visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bengaluru House Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price distribution
        axes[0, 0].hist(self.data['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price (Lakhs)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Area type distribution
        area_counts = self.data['area_type'].value_counts()
        axes[0, 1].pie(area_counts.values, labels=area_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Area Type Distribution')
        
        # 3. Total sqft vs Price
        axes[0, 2].scatter(self.data['total_sqft'], self.data['price'], alpha=0.6, color='green')
        axes[0, 2].set_title('Total Sqft vs Price')
        axes[0, 2].set_xlabel('Total Sqft')
        axes[0, 2].set_ylabel('Price (Lakhs)')
        
        # 4. Bathrooms vs Price
        bath_price = self.data.groupby('bath')['price'].mean()
        axes[1, 0].bar(bath_price.index, bath_price.values, color='orange', alpha=0.7)
        axes[1, 0].set_title('Average Price by Number of Bathrooms')
        axes[1, 0].set_xlabel('Number of Bathrooms')
        axes[1, 0].set_ylabel('Average Price (Lakhs)')
        
        # 5. Size distribution
        size_counts = self.data['size'].value_counts().head(10)
        axes[1, 1].barh(range(len(size_counts)), size_counts.values, color='purple', alpha=0.7)
        axes[1, 1].set_yticks(range(len(size_counts)))
        axes[1, 1].set_yticklabels(size_counts.index)
        axes[1, 1].set_title('Top 10 Size Categories')
        axes[1, 1].set_xlabel('Count')
        
        # 6. Correlation heatmap for numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional insights
        print(f"\nPrice Statistics:")
        print(f"Mean Price: {self.data['price'].mean():.2f} Lakhs")
        print(f"Median Price: {self.data['price'].median():.2f} Lakhs")
        print(f"Max Price: {self.data['price'].max():.2f} Lakhs")
        print(f"Min Price: {self.data['price'].min():.2f} Lakhs")
        
        print(f"\nMost expensive locations (top 10):")
        location_price = self.data.groupby('location')['price'].mean().sort_values(ascending=False).head(10)
        for loc, price in location_price.items():
            print(f"{loc}: {price:.2f} Lakhs")
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Create a copy for preprocessing
        df = self.data.copy()
        
        # 1. Handle missing values - Fix pandas warnings
        print("Handling missing values...")
        df.loc[:, 'society'] = df['society'].fillna('Unknown')
        df.loc[:, 'balcony'] = df['balcony'].fillna(df['balcony'].median())
        df.loc[:, 'bath'] = df['bath'].fillna(df['bath'].median())
        
        # 2. Clean total_sqft column - Improved function
        print("Cleaning total_sqft column...")
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
        
        # 3. Extract number of bedrooms from size
        print("Extracting bedroom count from size...")
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
        
        # 4. Handle availability
        print("Processing availability...")
        df['is_ready_to_move'] = (df['availability'] == 'Ready To Move').astype(int)
        
        # 5. Create location features
        print("Creating location features...")
        location_counts = df['location'].value_counts()
        df['location_frequency'] = df['location'].map(location_counts)
        
        # 6. Encode categorical variables
        print("Encoding categorical variables...")
        categorical_cols = ['area_type', 'location', 'society']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # 7. Select features for modeling
        feature_cols = [
            'total_sqft', 'bath', 'balcony', 'bedrooms', 'is_ready_to_move',
            'location_frequency', 'area_type_encoded', 'location_encoded'
        ]
        
        # Remove rows with missing values in features
        df_clean = df[feature_cols + ['price']].dropna()
        
        print(f"Final dataset shape after preprocessing: {df_clean.shape}")
        print(f"Features used: {feature_cols}")
        
        # Split the data
        X = df_clean[feature_cols]
        y = df_clean['price']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return df_clean
    
    def train_models(self):
        """Train multiple models and compare their performance"""
        print("\n=== MODEL TRAINING AND COMPARISON ===")
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        if XGBRegressor is not None:
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            # Use scaled data only for Linear Regression
            if name == 'Linear Regression':
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            model.fit(X_train_use, self.y_train)
            y_pred = model.predict(X_test_use)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring='r2')
            results[name] = {
                'model': model,
                'rmse': rmse,   
                'mae': mae,
                'r2': r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'predictions': y_pred
            }
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        self.models = results
        
        # Dynamic model selection based on actual performance
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        print(f"\n🏆 Best Model (based on R² score): {best_model_name}")
        print(f"📈 Best R² score: {results[best_model_name]['r2']:.4f}")
        print(f"💡 Note: Model selection is dynamic and based on actual performance, not assumptions")
        return results
    
    def plot_results(self, results):
        """Plot model comparison results"""
        print("\n=== MODEL COMPARISON VISUALIZATION ===")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. R² scores comparison
        model_names = list(results.keys())
        r2_scores = [results[name]['r2'] for name in model_names]
        
        bars = axes[0, 0].bar(model_names, r2_scores, color='lightcoral', alpha=0.7)
        axes[0, 0].set_title('R² Scores Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # 2. RMSE comparison
        rmse_scores = [results[name]['rmse'] for name in model_names]
        
        bars = axes[0, 1].bar(model_names, rmse_scores, color='lightblue', alpha=0.7)
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars, rmse_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{score:.1f}', ha='center', va='bottom')
        
        # 3. Actual vs Predicted for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        y_pred_best = results[best_model_name]['predictions']
        
        axes[1, 0].scatter(self.y_test, y_pred_best, alpha=0.6, color='green')
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Price')
        axes[1, 0].set_ylabel('Predicted Price')
        axes[1, 0].set_title(f'Actual vs Predicted ({best_model_name})')
        
        # 4. Cross-validation scores
        cv_means = [results[name]['cv_r2_mean'] for name in model_names]
        cv_stds = [results[name]['cv_r2_std'] for name in model_names]
        
        bars = axes[1, 1].bar(model_names, cv_means, yerr=cv_stds, 
                             color='lightgreen', alpha=0.7, capsize=5)
        axes[1, 1].set_title('Cross-Validation R² Scores')
        axes[1, 1].set_ylabel('CV R² Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars, cv_means):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed comparison table
        print("\nDetailed Model Comparison:")
        print("-" * 80)
        print(f"{'Model':<20} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'CV R²':<10}")
        print("-" * 80)
        
        for name in model_names:
            r2 = results[name]['r2']
            rmse = results[name]['rmse']
            mae = results[name]['mae']
            cv_r2 = results[name]['cv_r2_mean']
            print(f"{name:<20} {r2:<8.4f} {rmse:<8.2f} {mae:<8.2f} {cv_r2:<10.4f}")
    
    def feature_importance_analysis(self):
        """Analyze feature importance for tree-based models"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Get feature names
        feature_names = self.X_train.columns.tolist()
        
        # Analyze feature importance for tree-based models
        tree_models = ['Random Forest', 'Gradient Boosting']
        if XGBRegressor is not None:
            tree_models.append('XGBoost')
        
        fig, axes = plt.subplots(1, len(tree_models), figsize=(6*len(tree_models), 6))
        if len(tree_models) == 1:
            axes = [axes]
        
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        for i, model_name in enumerate(tree_models):
            if model_name in self.models:
                model = self.models[model_name]['model']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    axes[i].bar(range(len(importances)), importances[indices])
                    axes[i].set_title(f'{model_name} Feature Importance')
                    axes[i].set_xlabel('Features')
                    axes[i].set_ylabel('Importance')
                    axes[i].set_xticks(range(len(importances)))
                    axes[i].set_xticklabels([feature_names[j] for j in indices], rotation=45)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_sample(self, sample_data):
        """Make prediction for a sample house"""
        print("\n=== SAMPLE PREDICTION ===")
        sample_df = pd.DataFrame([sample_data])
        if 'size' in sample_df.columns:
            sample_df['bedrooms'] = sample_df['size'].apply(
                lambda x: int(x.split()[0]) if 'BHK' in str(x) else 
                         (int(x.split()[0]) if 'Bedroom' in str(x) else 1)
            )
        if 'availability' in sample_df.columns:
            sample_df['is_ready_to_move'] = (sample_df['availability'] == 'Ready To Move').astype(int)
        for col in ['area_type', 'location', 'society']:
            if col in sample_df.columns and col in self.label_encoders:
                sample_df[f'{col}_encoded'] = self.label_encoders[col].transform(
                    sample_df[col].astype(str)
                )
        if 'location' in sample_df.columns:
            location_counts = self.data['location'].value_counts()
            sample_df['location_frequency'] = sample_df['location'].map(location_counts)
        feature_cols = [
            'total_sqft', 'bath', 'balcony', 'bedrooms', 'is_ready_to_move',
            'location_frequency', 'area_type_encoded', 'location_encoded'
        ]
        sample_features = sample_df[feature_cols]
        sample_scaled = self.scaler.transform(sample_features)
        
        # Get predictions from all models
        predictions = {}
        print("Predictions from all models:")
        print("-" * 50)
        for name, result in self.models.items():
            if name == 'Linear Regression':
                pred = result['model'].predict(sample_scaled)[0]
            else:
                pred = result['model'].predict(sample_features)[0]
            predictions[name] = pred
            print(f"{name:<20}: ₹{pred:.2f} Lakhs")
        
        # Calculate prediction statistics
        pred_values = list(predictions.values())
        mean_pred = np.mean(pred_values)
        min_pred = min(pred_values)
        max_pred = max(pred_values)
        range_pred = max_pred - min_pred
        
        print("-" * 50)
        print(f"📊 Prediction Summary:")
        print(f"   Average: ₹{mean_pred:.2f} Lakhs")
        print(f"   Range: ₹{range_pred:.2f} Lakhs (₹{min_pred:.2f} - ₹{max_pred:.2f})")
        
        # Find most stable model (closest to mean)
        most_stable_model = min(predictions.keys(), key=lambda k: abs(predictions[k] - mean_pred))
        print(f"   Most Stable: {most_stable_model} (₹{predictions[most_stable_model]:.2f} Lakhs)")
        
        # Show best performing model (based on R²)
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['r2'])
        print(f"   Best R² Score: {most_stable_model} (R²: {self.models[best_model_name]['r2']:.3f})")
        
        print(f"\n💡 Model Selection Insights:")
        print(f"   • All models provide different perspectives on the property value")
        print(f"   • Range of ₹{range_pred:.2f} Lakhs shows market uncertainty")
        print(f"   • Consider the range when making investment decisions")
        
        return predictions[most_stable_model]  # Return most stable prediction
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("=== BENGALURU HOUSE PRICE PREDICTION ANALYSIS ===")
        print("=" * 60)
        
        # Load and explore data
        self.load_data()
        self.explore_data()
        
        # Preprocess data
        df_clean = self.preprocess_data()
        
        # Train models
        results = self.train_models()
        
        # Plot results
        self.plot_results(results)
        
        # Feature importance
        self.feature_importance_analysis()
        
        # Sample prediction
        sample_house = {
            'area_type': 'Super built-up  Area',
            'location': 'Whitefield',
            'size': '3 BHK',
            'total_sqft': 1500,
            'bath': 3,
            'balcony': 2,
            'availability': 'Ready To Move'
        }
        
        self.predict_sample(sample_house)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("Generated files:")
        print("- data_analysis.png: Exploratory data analysis plots")
        print("- model_comparison.png: Model performance comparison")
        print("- feature_importance.png: Feature importance analysis")
        
        return results

# Run the analysis
if __name__ == "__main__":
    predictor = HousePricePredictor('Bengaluru_House_Data.csv')
    results = predictor.run_complete_analysis() 