# detector/views.py
from django.shortcuts import render
from django.http import JsonResponse
# Import the custom model loader module
from .model_loader import BEST_MODEL, VECTORIZER, translate_arabic_to_english, clean_english_text, TRANSLATOR_MODEL 
import json

def detector_home(request):
    """Renders the main input page."""
    return render(request, 'detector/home.html')

def predict_api(request):
    """Handles the prediction logic (POST request)."""
    if request.method == 'POST':
        # 1. Check Model Status
        if not BEST_MODEL or not VECTORIZER or not TRANSLATOR_MODEL:
            return JsonResponse({'error': 'Model not yet initialized or failed to load. Check server logs.'}, status=500)

        try:
            # Get user input from the JSON body of the POST request
            data = json.loads(request.body)
            arabic_input = data.get('text', '')
            
            if not arabic_input:
                return JsonResponse({'error': 'No text provided.'}, status=400)

            # 2. Translation
            english_translated = translate_arabic_to_english(arabic_input)

            # 3. Preprocessing
            cleaned_text = clean_english_text(english_translated)

            # 4. Feature Vectorization
            # The vectorizer must be fed a list/iterable of documents
            X_test = VECTORIZER.transform([cleaned_text])

            # 5. Prediction
            prediction = BEST_MODEL.predict(X_test)[0]
            
            # 6. Format Output
            result = {
                'original_arabic': arabic_input,
                'translated_english': english_translated,
                'prediction_label': int(prediction),
                'prediction_text': 'FAKE' if prediction == 1 else 'REAL',
                'model_name': 'SVC (F1: 0.9958)'
            }
            
            return JsonResponse(result)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format. Please send text in the request body.'}, status=400)
        except Exception as e:
            print(f"Prediction Error: {e}")
            return JsonResponse({'error': f'An internal server error occurred during prediction: {str(e)}'}, status=500)
            
    return JsonResponse({'error': 'Only POST method is allowed for prediction.'}, status=405)