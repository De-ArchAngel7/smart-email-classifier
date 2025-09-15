from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
from typing import Dict, Any, List
import re

app = FastAPI(title="Smart Email Classifier", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
try:
    classifier = pipeline(
        "text-classification", 
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        top_k=None
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

class EmailRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    category: str
    confidence: float
    all_scores: list

# Map emotion labels to email categories
EMOTION_TO_CATEGORY = {
    "joy": "Positive Feedback",
    "sadness": "Refund",
    "anger": "Complaint",
    "fear": "Technical Issue",
    "surprise": "General Inquiry",
    "love": "Positive Feedback"
}

def map_emotion_to_category(emotion: str) -> str:
    """Map emotion classification to email category"""
    return EMOTION_TO_CATEGORY.get(emotion, "General Inquiry")

def split_text_into_chunks(text: str, max_length: int = 400) -> List[str]:
    """Split text into overlapping chunks for better classification"""
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length > max_length and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap (last 50 words)
            overlap_size = min(50, len(current_chunk))
            current_chunk = current_chunk[-overlap_size:] + [word]
            current_length = len(' '.join(current_chunk))
        else:
            current_chunk.append(word)
            current_length += word_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def combine_classification_results(chunk_results: List[List[Dict]]) -> Dict:
    """Combine results from multiple chunks using weighted averaging"""
    if not chunk_results:
        # Return a default result when no chunks are provided
        return {
            'category': 'General Inquiry',
            'confidence': 0.0,
            'all_scores': [{'emotion': 'surprise', 'category': 'General Inquiry', 'confidence': 0.0}]
        }
    
    # Flatten all results
    all_scores = []
    for chunk_result in chunk_results:
        all_scores.extend(chunk_result)
    
    # Group by emotion and calculate weighted average
    emotion_scores = {}
    total_weight = 0
    
    for result in all_scores:
        emotion = result['label']
        score = result['score']
        
        if emotion not in emotion_scores:
            emotion_scores[emotion] = {'total_score': 0, 'count': 0}
        
        emotion_scores[emotion]['total_score'] += score
        emotion_scores[emotion]['count'] += 1
        total_weight += score
    
    # Calculate average scores
    combined_scores = []
    for emotion, data in emotion_scores.items():
        avg_score = data['total_score'] / data['count']
        combined_scores.append({
            'emotion': emotion,
            'category': map_emotion_to_category(emotion),
            'confidence': avg_score
        })
    
    # Sort by confidence
    combined_scores.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Get the best result
    best_result = combined_scores[0]
    
    return {
        'category': best_result['category'],
        'confidence': best_result['confidence'],
        'all_scores': combined_scores
    }

@app.get("/")
async def root():
    return {"message": "Smart Email Classifier API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": classifier is not None}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_email(request: EmailRequest):
    if not classifier:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty")
    
    try:
        # Split text into chunks for better processing
        chunks = split_text_into_chunks(request.text, max_length=400)
        
        # Classify each chunk
        chunk_results = []
        for chunk in chunks:
            chunk_result = classifier(chunk)
            chunk_results.append(chunk_result[0])
        
        # Combine results from all chunks
        combined_result = combine_classification_results(chunk_results)
        
        return ClassificationResponse(
            category=combined_result['category'],
            confidence=combined_result['confidence'],
            all_scores=combined_result['all_scores']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
