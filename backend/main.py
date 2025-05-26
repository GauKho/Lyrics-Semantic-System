from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from retrieval.hybrid_lyrics_searching import HybridLyricsSearch
from pydantic import BaseModel
from typing import List

app = FastAPI()

DATA_PATH = "backend\data\csv"

class Model(BaseModel):
    query: str 
    top_k: int

@app.get("/")
async def read_root():
        return {f"This is Lyrics Semantic Searching"}

@app.get("/search_lyrics", response_class=List[Model])
async def searching_lyrics(request):