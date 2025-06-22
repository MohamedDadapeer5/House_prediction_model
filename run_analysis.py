#!/usr/bin/env python3
"""
Simple script to run the house price prediction analysis with dynamic model selection
"""

from house_price_prediction import HousePricePredictor

def main():
    print("🚀 Starting Bengaluru House Price Prediction Analysis...")
    print("=" * 60)
    print("🎯 Using Dynamic Model Selection (No Hardcoded Bias)")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = HousePricePredictor('Bengaluru_House_Data.csv')
        
        # Run complete analysis
        results = predictor.run_complete_analysis()
        
        print("\n✅ Analysis completed successfully!")
        print("\n📊 Summary of Results:")
        print("-" * 40)
        
        # Show best model based on R² score
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_result = results[best_model_name]
        
        print(f"🏆 Best R² Score: {best_model_name}")
        print(f"📈 R² Score: {best_result['r2']:.4f}")
        print(f"📉 RMSE: {best_result['rmse']:.2f}")
        print(f"📊 MAE: {best_result['mae']:.2f}")
        
        print(f"\n💡 Model Selection Philosophy:")
        print(f"   • All models are evaluated fairly based on actual performance")
        print(f"   • No hardcoded assumptions about which model is 'best'")
        print(f"   • Dynamic selection based on prediction stability")
        print(f"   • Users see all predictions for informed decisions")
        
        # Test sample prediction
        print("\n🧪 Testing Sample Prediction:")
        print("-" * 40)
        
        sample_house = {
            'area_type': 'Super built-up  Area',
            'location': 'Whitefield',
            'size': '3 BHK',
            'total_sqft': 1500,
            'bath': 3,
            'balcony': 2,
            'availability': 'Ready To Move'
        }
        
        print("Sample Property Details:")
        for key, value in sample_house.items():
            print(f"  {key}: {value}")
        
        predicted_price = predictor.predict_sample(sample_house)
        print(f"\n💰 Most Stable Prediction: ₹{predicted_price:.2f} Lakhs")
        print(f"💡 Note: This is the model prediction closest to the average of all models")
        
        print("\n🎉 All done! Check the generated files:")
        print("  - data_analysis.png")
        print("  - model_comparison.png") 
        print("  - feature_importance.png")
        
        print(f"\n🌐 To start the web application with dynamic model selection:")
        print(f"   python app.py")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please make sure the dataset file 'Bengaluru_House_Data.csv' is in the current directory.")

if __name__ == "__main__":
    main() 