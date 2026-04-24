from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
from typing import List, Dict, Any
import uvicorn

# Import our custom modules
from email_classify import email_extract
from IDS import predict_from_csv

app = FastAPI(
    title="AI Threat Intelligence API",
    description="API for email phishing detection and network intrusion detection",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Threat Intelligence API",
        "endpoints": {
            "/email-classify": "GET - Extract and classify emails for phishing detection",
            "/ids-predict": "POST - Upload CSV file for network intrusion detection"
        }
    }

@app.get("/email-classify")
async def classify_emails():
    """
    Extract unseen emails from Gmail and classify them for phishing detection.
    Returns: List of emails with sender info and classification status.
    """
    try:
        results = email_extract()
        return {
            "status": "success",
            "message": f"Processed {len(results)} emails",
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing emails: {str(e)}")

@app.post("/ids-predict")
async def predict_intrusion(file: UploadFile = File(...)):
    """
    Upload a CSV file for network intrusion detection.
    Accepts: CSV file with network traffic data
    Returns: Predictions for each row in the CSV
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        csv_string = contents.decode('utf-8')
        csv_file = io.StringIO(csv_string)
        
        # Get predictions using our IDS module
        predictions = predict_from_csv(csv_file)
        
        # Convert predictions to list for JSON serialization
        predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
        
        return {
            "status": "success",
            "message": f"Processed {len(predictions_list)} records",
            "filename": file.filename,
            "predictions": predictions_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Threat Intelligence API"}

if __name__ == "__main__":
    print("Starting AI Threat Intelligence API...")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
