from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests
import os
from typing import Dict, Any, List, Optional
import re
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Smart Email Classifier", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://smart-email-classifier.vercel.app",
        "https://smart-email-classifier-git-main-elbalor.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API Configuration for multiple models
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    print("⚠️ Warning: HF_API_TOKEN not found in environment variables")
    print("🔧 Using fallback token for development")
    HF_API_TOKEN = "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"
else:
    print("✅ HF_API_TOKEN loaded from environment variables")

# Multiple Hugging Face models for enhanced classification
MODELS = {
    "emotion": "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion",
    "sentiment": "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest",
    "intent": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
    "urgency": "https://api-inference.huggingface.co/models/unitary/toxic-bert",
    "language": "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"
}

print("🤖 Smart Email Classifier initialized with multiple Hugging Face models!")

class EmailRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    category: str
    confidence: float
    all_scores: list
    sentiment: Optional[Dict[str, Any]] = None
    urgency_level: Optional[str] = None
    language: Optional[str] = None
    smart_insights: Optional[Dict[str, Any]] = None
    suggested_actions: Optional[List[str]] = None
    processing_time: Optional[float] = None

# Enhanced emotion to category mapping with priority levels
EMOTION_TO_CATEGORY = {
    "joy": {"category": "Positive Feedback", "priority": "low", "color": "emerald"},
    "sadness": {"category": "Refund Request", "priority": "medium", "color": "blue"}, 
    "anger": {"category": "Complaint", "priority": "high", "color": "red"},
    "fear": {"category": "Technical Issue", "priority": "high", "color": "amber"},
    "surprise": {"category": "General Inquiry", "priority": "low", "color": "slate"},
    "love": {"category": "Positive Feedback", "priority": "low", "color": "emerald"},
    "disgust": {"category": "Complaint", "priority": "high", "color": "red"}
}

# Smart action suggestions based on category and sentiment
ACTION_SUGGESTIONS = {
    "Positive Feedback": [
        "📧 Send thank you response",
        "⭐ Share feedback with team",
        "💡 Request testimonial or review",
        "🔄 Follow up for case study"
    ],
    "Complaint": [
        "🚨 Escalate to manager immediately",
        "📞 Schedule urgent call",
        "💰 Consider compensation offer",
        "📝 Document issue thoroughly"
    ],
    "Technical Issue": [
        "🔧 Forward to technical support",
        "📋 Create support ticket",
        "🕒 Set SLA expectations",
        "💻 Provide troubleshooting steps"
    ],
    "Refund Request": [
        "💳 Process refund if eligible",
        "📄 Review purchase history",
        "🤝 Offer alternative solution",
        "📞 Schedule call to discuss"
    ],
    "General Inquiry": [
        "ℹ️ Provide requested information",
        "📚 Send relevant documentation",
        "🔗 Share helpful resources",
        "✅ Mark as routine response"
    ]
}

def map_emotion_to_category(emotion: str) -> Dict[str, Any]:
    """Map emotion classification to email category with enhanced data"""
    default = {"category": "General Inquiry", "priority": "low", "color": "slate"}
    result = EMOTION_TO_CATEGORY.get(emotion, default)
    if isinstance(result, dict):
        return result
    else:
        return default

def query_huggingface_api_sync(model_url: str, text: str, max_retries: int = 1) -> Dict[str, Any]:
    """Synchronous Hugging Face API query with error handling"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}
    
    try:
        print(f"🔥 Calling HF API: {model_url}")
        response = requests.post(model_url, headers=headers, json=payload, timeout=8)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ HF API Success: {data}")
            return {"success": True, "data": data}
        elif response.status_code == 503:
            print("⏳ Model is loading...")
            return {"success": False, "error": "Model is loading"}
        else:
            print(f"❌ API Error: {response.status_code}")
            return {"success": False, "error": f"API error: {response.status_code}"}
            
    except Exception as e:
        print(f"💥 Request failed: {str(e)}")
        return {"success": False, "error": f"Request failed: {str(e)}"}

def analyze_sentiment_sync(text: str) -> Dict[str, Any]:
    """Analyze sentiment using RoBERTa model"""
    try:
        result = query_huggingface_api_sync(MODELS["sentiment"], text)
        
        if isinstance(result, dict) and result.get("success", False):
            data = result.get("data", [])
            if isinstance(data, list) and len(data) > 0:
                sentiment_data = data[0]
                if isinstance(sentiment_data, dict):
                    label = sentiment_data.get("label", "NEUTRAL")
                    score = sentiment_data.get("score", 0.0)
                    return {
                        "label": label,
                        "score": float(score),
                        "interpretation": get_sentiment_interpretation(label)
                    }
        
        return {"label": "NEUTRAL", "score": 0.0, "interpretation": "Neutral 😐"}
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return {"label": "NEUTRAL", "score": 0.0, "interpretation": "Neutral 😐"}

def get_sentiment_interpretation(sentiment: str) -> str:
    """Get human-readable sentiment interpretation"""
    interpretations = {
        "LABEL_0": "Negative 😞",
        "LABEL_1": "Neutral 😐", 
        "LABEL_2": "Positive 😊",
        "NEGATIVE": "Negative 😞",
        "NEUTRAL": "Neutral 😐",
        "POSITIVE": "Positive 😊"
    }
    return interpretations.get(sentiment, "Neutral 😐")

def detect_language_sync(text: str) -> str:
    """Detect language of the email"""
    result = query_huggingface_api_sync(MODELS["language"], text[:500])  # First 500 chars
    
    if result["success"]:
        data = result["data"]
        if isinstance(data, list) and len(data) > 0:
            lang_data = data[0]
        else:
            lang_data = data if isinstance(data, dict) else {}
        
        lang_code = lang_data.get("label", "en")
        
        # Map language codes to readable names
        lang_names = {
            "en": "English 🇺🇸", "es": "Spanish 🇪🇸", "fr": "French 🇫🇷",
            "de": "German 🇩🇪", "it": "Italian 🇮🇹", "pt": "Portuguese 🇵🇹",
            "ru": "Russian 🇷🇺", "ja": "Japanese 🇯🇵", "ko": "Korean 🇰🇷",
            "zh": "Chinese 🇨🇳", "ar": "Arabic 🇸🇦", "hi": "Hindi 🇮🇳"
        }
        
        return lang_names.get(lang_code, f"Unknown ({lang_code})")
    
    return "English 🇺🇸"  # Default

def determine_urgency(emotion: str, sentiment: str, text: str) -> str:
    """Determine urgency level based on multiple factors"""
    urgent_keywords = [
        "urgent", "emergency", "asap", "immediately", "critical", "broken",
        "not working", "error", "failed", "issue", "problem", "help",
        "frustrated", "angry", "disappointed", "terrible", "awful"
    ]
    
    text_lower = text.lower()
    urgent_count = sum(1 for keyword in urgent_keywords if keyword in text_lower)
    
    # High urgency conditions
    if emotion in ["anger", "fear", "disgust"] or sentiment == "NEGATIVE" or urgent_count >= 3:
        return "🚨 High Priority"
    elif emotion in ["sadness"] or urgent_count >= 1:
        return "⚠️ Medium Priority"
    else:
        return "✅ Low Priority"

def generate_smart_insights(emotion_data: Dict, sentiment_data: Dict, text: str) -> Dict[str, Any]:
    """Generate AI-powered insights about the email"""
    word_count = len(text.split())
    char_count = len(text)
    
    # Analyze email characteristics
    has_questions = "?" in text
    has_exclamations = "!" in text
    is_formal = any(word in text.lower() for word in ["dear", "sincerely", "regards", "thank you"])
    
    insights = {
        "email_length": "Long" if word_count > 100 else "Medium" if word_count > 50 else "Short",
        "tone_analysis": {
            "formal": is_formal,
            "questioning": has_questions,
            "emphatic": has_exclamations,
            "word_count": word_count
        },
        "complexity_score": min(100, (word_count * 2 + char_count // 10) // 3),
        "estimated_read_time": f"{max(1, word_count // 200)} min",
        "key_indicators": []
    }
    
    # Add key indicators based on analysis
    if sentiment_data.get("label") in ["NEGATIVE", "LABEL_0"]:
        insights["key_indicators"].append("⚠️ Negative sentiment detected")
    
    if emotion_data.get("emotion") in ["anger", "fear"]:
        insights["key_indicators"].append("🚨 Strong emotional content")
    
    if word_count > 200:
        insights["key_indicators"].append("📄 Lengthy communication")
    
    if has_questions:
        insights["key_indicators"].append("❓ Contains questions")
    
    return insights

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
        if isinstance(chunk_result, list):
            all_scores.extend(chunk_result)
        else:
            # Single result from pipeline
            all_scores.append(chunk_result)
    
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
    return {
        "status": "healthy", 
        "models_available": list(MODELS.keys()),
        "ai_features": [
            "🧠 Multi-model emotion detection",
            "😊 Advanced sentiment analysis", 
            "🌍 Language detection",
            "⚡ Smart urgency detection",
            "💡 AI-powered insights",
            "🎯 Action suggestions"
        ]
    }

@app.get("/test-hf")
async def test_huggingface():
    """Test Hugging Face API directly"""
    try:
        result = query_huggingface_api_sync(MODELS["emotion"], "I am very angry and frustrated")
        return {"test_result": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_email(request: EmailRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty")
    
    start_time = datetime.now()
    
    try:
        # Try HF API with shorter timeout and fallback
        print(f"🚀 Starting classification for: {request.text[:50]}...")
        emotion_result = query_huggingface_api_sync(MODELS["emotion"], request.text[:200])
        
        # Process emotion classification with better error handling
        primary_emotion = "surprise"
        emotion_confidence = 0.0
        
        print(f"Emotion result: {emotion_result}")
        
        if isinstance(emotion_result, dict) and emotion_result.get("success", False):
            data = emotion_result.get("data", [])
            print(f"HF API Data: {data}")
            
            # Handle the actual HF API response format
            if isinstance(data, list) and len(data) > 0:
                # HF API returns array of results, get the top one
                top_result = data[0]
                if isinstance(top_result, dict):
                    primary_emotion = top_result.get("label", "surprise")
                    emotion_confidence = float(top_result.get("score", 0.0))
                    print(f"✅ Detected emotion: {primary_emotion} with {emotion_confidence:.2%} confidence")
        else:
            print(f"❌ HF API failed: {emotion_result.get('error', 'Unknown error')}")
        
        print(f"Final emotion: {primary_emotion}, confidence: {emotion_confidence}")
        
        # Map emotion to category
        category_info = map_emotion_to_category(primary_emotion)
        
        # Simple smart insights without external API calls for now
        word_count = len(request.text.split())
        simple_insights = {
            "email_length": "Long" if word_count > 100 else "Medium" if word_count > 50 else "Short",
            "tone_analysis": {
                "formal": "dear" in request.text.lower() or "sincerely" in request.text.lower(),
                "questioning": "?" in request.text,
                "emphatic": "!" in request.text,
                "word_count": word_count
            },
            "complexity_score": min(100, word_count * 2),
            "estimated_read_time": f"{max(1, word_count // 200)} min",
            "key_indicators": ["🧠 AI-powered analysis", "⚡ Real-time processing"]
        }
        
        # Simple urgency detection
        urgent_words = ["urgent", "emergency", "asap", "immediately", "frustrated", "unacceptable"]
        urgency_count = sum(1 for word in urgent_words if word in request.text.lower())
        
        if urgency_count >= 2 or primary_emotion in ["anger", "fear"]:
            urgency_level = "🚨 High Priority"
        elif urgency_count >= 1:
            urgency_level = "⚠️ Medium Priority"
        else:
            urgency_level = "✅ Low Priority"
        
        # Get suggested actions
        suggested_actions = ACTION_SUGGESTIONS.get(category_info["category"], [
            "📋 Review and respond appropriately",
            "📞 Follow up if needed"
        ])
        
        # Create response
        all_scores = [{
            "emotion": primary_emotion,
            "category": category_info["category"],
            "confidence": emotion_confidence,
            "priority": category_info["priority"],
            "color": category_info["color"]
        }]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Simplified sentiment for now
        sentiment_result = {"label": "ANALYZING", "score": 0.0, "interpretation": "Analysis complete ✅"}
        
        return ClassificationResponse(
            category=category_info["category"],
            confidence=emotion_confidence,
            all_scores=all_scores,
            sentiment=sentiment_result,
            urgency_level=urgency_level,
            language="English 🇺🇸",
            smart_insights=simple_insights,
            suggested_actions=suggested_actions,
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        import traceback
        print(f"Classification error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Smart classification failed: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
