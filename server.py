import warnings
warnings.filterwarnings("ignore")

import time
import io
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image

from fastapi.responses import FileResponse
import uuid
from src.pdf_to_docx_api import convert_pdf_to_docx


# Get the directory of this file
BASE_DIR = Path(__file__).parent
FONTS_DIR = BASE_DIR / "fonts"
MODELS_DIR = BASE_DIR / "models"
LANGUAGES_FILE = BASE_DIR / "languages.json"

# Ensure directories exist
FONTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Load language configurations
with open(LANGUAGES_FILE, 'r', encoding='utf-8') as f:
    languages_config = json.load(f)

# Initialize OCR models cache
ocr_models: Dict[str, PaddleOCR] = {}

def get_ocr_model(lang_code: str) -> PaddleOCR:
    """Get or create OCR model for the specified language."""
    if lang_code not in ocr_models:
        # Initialize PaddleOCR (version 3.x API)
        ocr_models[lang_code] = PaddleOCR(
            lang=lang_code,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    return ocr_models[lang_code]

app = FastAPI(
    title="OCR Service",
    description="Text extraction service using PaddleOCR",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "OCR Text Extraction Service",
        "version": "1.0.0",
        "supported_languages": list(languages_config["languages"].keys()),
        "endpoints": {
            "extract": "/extract_text/",
            "languages": "/languages/",
            "health": "/health/"
        }
    }

@app.get("/languages/")
async def get_languages():
    """Get list of supported languages."""
    return {
        "languages": languages_config["languages"],
        "total": len(languages_config["languages"])
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(ocr_models),
        "available_languages": list(languages_config["languages"].keys())
    }

@app.post("/extract_text/")
async def extract_text(
    file: UploadFile = File(...),
    lang: str = Query("en", description="Language code (e.g., en, hi, ta, te, kn, ml, bn, mr)"),
    min_confidence: float = Query(0.7, description="Minimum confidence threshold (0.0 to 1.0)", ge=0.0, le=1.0)
):
    """
    Extract text from an uploaded image using OCR.
    
    Parameters:
    - file: Image file (PNG, JPG, JPEG, etc.)
    - lang: Language code for OCR (default: en)
    - min_confidence: Minimum confidence threshold for text detection (default: 0.7)
    
    Returns:
    - extracted_text: Full extracted text as a single string
    - details: List of detected text blocks with coordinates and confidence scores
    - metadata: Processing information
    """
    start_time = time.time()
    
    # Validate language
    if lang not in languages_config["languages"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {lang}. Supported languages: {list(languages_config['languages'].keys())}"
        )
    
    try:
        # Read uploaded file into PIL Image
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Convert PIL -> numpy for PaddleOCR
        np_image = np.array(pil_image)
        
        # Get OCR model for the language
        ocr_model = get_ocr_model(lang)
        
        # Run OCR using predict method (PaddleOCR 3.x)
        results = ocr_model.predict(np_image)
        
        if not results or len(results) == 0:
            return JSONResponse(content={
                "extracted_text": "",
                "details": [],
                "metadata": {
                    "language": lang,
                    "language_name": languages_config["languages"][lang]["name"],
                    "image_size": {"width": pil_image.width, "height": pil_image.height},
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                    "total_detections": 0
                }
            })
        
        # Extract text and details from results
        text_blocks = []
        full_text_lines = []
        
        # PaddleOCR 3.x returns a list with dictionary results for each page
        page = results[0]
        rec_texts = page['rec_texts']
        rec_scores = page['rec_scores']
        rec_polys = page['rec_polys']
        
        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            # Filter by confidence threshold
            if score >= min_confidence:
                full_text_lines.append(text)
                
                # Convert poly to the expected format
                text_blocks.append({
                    "text": text,
                    "confidence": round(float(score), 4),
                    "bounding_box": {
                        "top_left": [float(poly[0][0]), float(poly[0][1])],
                        "top_right": [float(poly[1][0]), float(poly[1][1])],
                        "bottom_right": [float(poly[2][0]), float(poly[2][1])],
                        "bottom_left": [float(poly[3][0]), float(poly[3][1])]
                    }
                })
        
        # Combine all text
        extracted_text = "\n".join(full_text_lines)
        
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        
        return JSONResponse(content={
            "extracted_text": extracted_text,
            "details": text_blocks,
            "metadata": {
                "language": lang,
                "language_name": languages_config["languages"][lang]["name"],
                "image_size": {"width": pil_image.width, "height": pil_image.height},
                "processing_time_ms": elapsed_ms,
                "total_detections": len(text_blocks),
                "confidence_threshold": min_confidence
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

@app.post("/convert/pdf-to-docx/")
async def pdf_to_docx_api(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_id = str(uuid.uuid4())
    pdf_path = UPLOADS_DIR / f"{file_id}.pdf"
    docx_path = OUTPUTS_DIR / f"{file_id}.docx"

    # Save uploaded PDF
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    # Convert PDF → DOCX
    convert_pdf_to_docx(str(pdf_path), str(docx_path))

    return FileResponse(
        path=docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="converted.docx"
    )
