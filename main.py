import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from hashlib import sha256

from database import db, create_document, get_documents
from schemas import Generation

app = FastAPI(title="YOURSELF API", description="Ultra-realistic model prompt enhancer and generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EnhanceRequest(BaseModel):
    prompt: str = Field(..., description="User base prompt")
    age: Optional[str] = None
    skin_tone: Optional[str] = None
    eye_color: Optional[str] = None
    nationality: Optional[str] = None

class EnhanceResponse(BaseModel):
    enhanced_prompt: str

class GenerateRequest(EnhanceRequest):
    enhanced_prompt: Optional[str] = None

class GenerateResponse(BaseModel):
    image_url: str
    enhanced_prompt: str
    id: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "YOURSELF backend running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from YOURSELF API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


def build_enhanced_prompt(req: EnhanceRequest) -> str:
    base = req.prompt.strip()
    if not base:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    details = []
    if req.age:
        details.append(f"age: {req.age}")
    if req.skin_tone:
        details.append(f"skin tone: {req.skin_tone}")
    if req.eye_color:
        details.append(f"eye color: {req.eye_color}")
    if req.nationality:
        details.append(f"nationality: {req.nationality}")

    guide = (
        "ultra-realistic portrait, 85mm lens, shallow depth of field, soft natural lighting, high dynamic range, "
        "detailed skin texture, cinematic color grading, volumetric light, subsurface scattering, 8k, photorealistic"
    )

    attributes = ", ".join(details) if details else ""
    enhanced = f"{base}. {attributes}. {guide}. head-and-shoulders composition, looking at camera"
    return " ".join(enhanced.split())


@app.post("/api/enhance", response_model=EnhanceResponse)
def enhance(req: EnhanceRequest):
    enhanced_prompt = build_enhanced_prompt(req)
    return {"enhanced_prompt": enhanced_prompt}


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # Build or use enhanced prompt
    enhanced_prompt = req.enhanced_prompt or build_enhanced_prompt(req)

    # Demo image generation strategy: use deterministic placeholder from picsum with a seed
    seed = sha256(enhanced_prompt.encode("utf-8")).hexdigest()[:16]
    width = 768
    height = 1024
    # Using picsum seeded image as placeholder; in production, call a real image gen provider here
    image_url = f"https://picsum.photos/seed/{seed}/{width}/{height}"

    # Persist to database
    try:
        gen = Generation(
            prompt=req.prompt,
            enhanced_prompt=enhanced_prompt,
            age=req.age,
            skin_tone=req.skin_tone,
            eye_color=req.eye_color,
            nationality=req.nationality,
            image_url=image_url,
        )
        _id = create_document("generation", gen)
    except Exception:
        _id = None  # If DB not available, still return result

    return {"image_url": image_url, "enhanced_prompt": enhanced_prompt, "id": _id}


@app.get("/api/generations")
def list_generations(limit: int = 20):
    try:
        docs = get_documents("generation", {}, limit)
        # Convert ObjectId to str if present
        out = []
        for d in docs:
            d["id"] = str(d.pop("_id", ""))
            out.append(d)
        return {"items": out}
    except Exception as e:
        # If DB not available, return empty list
        return {"items": []}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
