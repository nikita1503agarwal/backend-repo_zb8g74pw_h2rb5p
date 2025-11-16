import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from hashlib import sha256
import requests
import logging
from datetime import datetime

from database import db, create_document, get_documents
from schemas import Generation

# Configure basic logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    source: Literal["stability", "fallback"]
    seed: Optional[int] = None
    error: Optional[str] = None


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
    response["stability_key"] = "✅ Set" if os.getenv("STABILITY_API_KEY") else "❌ Not Set"
    return response


def coerce_adult_age(age: Optional[str]) -> str:
    # Ensure we always prompt for an adult subject
    if not age:
        return "mid 20s"
    try:
        # If numeric, ensure >= 20
        n = int(''.join([c for c in age if c.isdigit()]))
        return f"{max(n, 20)}"
    except Exception:
        # textual ages like 'early 20s' pass through if adult-ish, else coerce
        text = age.lower()
        if any(k in text for k in ["teen", "child", "kid", "girl", "boy", "minor", "under"]):
            return "mid 20s"
        return age


DISALLOWED_TOKENS = [
    "text", "watermark", "logo", "fruit", "cherries", "palace", "castle", "building", "landscape",
    "extra limbs", "distortion", "deformed", "out of frame", "nsfw", "nude", "child", "minor", "kid",
    "cartoon", "illustration", "painting", "render"
]


def sanitize_prompt(base: str) -> str:
    b = base
    low = b.lower()
    for token in DISALLOWED_TOKENS:
        if token in low:
            # remove the span loosely by replacing token (case-insensitive)
            b = " ".join(part for part in b.split() if token.lower() not in part.lower())
            low = b.lower()
    return " ".join(b.split())


def build_enhanced_prompt(req: EnhanceRequest) -> str:
    raw = req.prompt.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Remove disallowed/derailing terms from user prompt
    base = sanitize_prompt(raw)

    # Strengthen subject to avoid random objects/backgrounds and ensure adult subject
    age_text = coerce_adult_age(req.age)

    # Determine subject prefix, enforcing adult
    subj_prefix = "solo portrait of an adult woman"
    base_lower = base.lower()
    if any(k in base_lower for k in ["man", "male", "boy", "guy"]):
        subj_prefix = "solo portrait of an adult man"
    if any(k in base_lower for k in ["woman", "female", "lady"]):
        subj_prefix = "solo portrait of an adult woman"
    if any(k in base_lower for k in ["girl", "boy", "kid", "teen"]):
        subj_prefix = "solo portrait of an adult woman" if "girl" in base_lower else "solo portrait of an adult man"

    details = []
    if age_text:
        details.append(f"age: {age_text}")
    if req.skin_tone:
        details.append(f"skin tone: {req.skin_tone}")
    if req.eye_color:
        details.append(f"eye color: {req.eye_color}")
    if req.nationality:
        details.append(f"nationality: {req.nationality}")

    guide = (
        "ultra-realistic photographic portrait, 85mm lens, shallow depth of field, soft natural lighting, high dynamic range, "
        "detailed skin texture, cinematic color grading, volumetric light, subsurface scattering, 8k, photorealistic, "
        "upper body, centered composition, looking at camera, neutral studio background, face in focus, sharp eyes"
    )

    disallow = (
        "no text, no watermark, no logo, no fruit, no cherries, no palace, no castle, no building, no landscape, "
        "no extra limbs, no distortion, no deformed face, no out of frame, no cartoon, no painting, no render"
    )

    attributes = ", ".join(details) if details else ""
    enhanced = f"{subj_prefix}, {base}. {attributes}. {guide}. {disallow}"
    enhanced = " ".join(enhanced.split())
    logger.info("Enhanced prompt built: %s", enhanced)
    return enhanced


def prompt_seed(text: str) -> int:
    # Deterministic seed from text within 32-bit range
    return int(sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def generate_with_stability(enhanced_prompt: str, width: int = 768, height: int = 1024) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Generate an image using Stability AI REST API. Returns (data URL, seed, error)."""
    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        return None, None, "STABILITY_API_KEY not set"
    try:
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        negative = (
            "low quality, blurry, distorted, deformed, extra fingers, watermark, text, logo, nsfw, nude, child, minor, kid, toddler, baby, "
            "cartoon, illustration, painting, 3d render, landscape, palace, castle, fruit, cherries, bad anatomy, duplicate"
        )
        seed = prompt_seed(enhanced_prompt)
        payload = {
            "text_prompts": [
                {"text": enhanced_prompt, "weight": 1.2},
                {"text": negative, "weight": -1.6},
            ],
            "cfg_scale": 8,
            "clip_guidance_preset": "FAST_BLUE",
            "sampler": "K_DPM_2_ANCESTRAL",
            "samples": 1,
            "steps": 40,
            "width": width,
            "height": height,
            "seed": seed,
            "style_preset": "photographic",
        }
        logger.info("Stability request | seed=%s | size=%sx%s", seed, width, height)
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            logger.warning("Stability error status: %s | body=%s", resp.status_code, resp.text[:300])
            return None, seed, f"HTTP {resp.status_code}"
        data = resp.json()
        artifacts = data.get("artifacts") or []
        if not artifacts:
            logger.warning("Stability response has no artifacts")
            return None, seed, "No artifacts"
        b64 = artifacts[0].get("base64")
        if not b64:
            logger.warning("Stability response missing base64")
            return None, seed, "No base64"
        return f"data:image/png;base64,{b64}", seed, None
    except Exception as e:
        logger.exception("Stability request failed: %s", e)
        return None, None, str(e)


@app.post("/api/enhance", response_model=EnhanceResponse)
def enhance(req: EnhanceRequest):
    enhanced_prompt = build_enhanced_prompt(req)
    return {"enhanced_prompt": enhanced_prompt}


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # Build or use enhanced prompt
    enhanced_prompt = req.enhanced_prompt or build_enhanced_prompt(req)

    # Try real image generation first (Stability AI). Falls back to placeholder if not configured.
    width, height = 768, 1024
    img, seed, err = generate_with_stability(enhanced_prompt, width=width, height=height)

    source = "stability" if img else "fallback"
    if not img:
        # Fallback deterministic placeholder from picsum with a seed
        seed_hex = sha256(enhanced_prompt.encode("utf-8")).hexdigest()[:16]
        image_url = f"https://picsum.photos/seed/{seed_hex}/{width}/{height}"
    else:
        image_url = img

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
    except Exception as e:
        logger.warning("DB save failed: %s", e)
        _id = None  # If DB not available, still return result

    logger.info("Generate completed | source=%s | seed=%s", source, seed)
    return {"image_url": image_url, "enhanced_prompt": enhanced_prompt, "id": _id, "source": source, "seed": seed, "error": err}


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
