from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import time # Import time for timestamp comparison

app = FastAPI(title="Intelligence Command Center")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RoomData(BaseModel):
    room_id: str
    is_occupied: bool
    temperature: float
    light: float
    humidity: float
    pir: int

internal_db: Dict[str, dict] = {}

@app.post("/update")
async def receive_update(data: RoomData):
    node_packet = data.dict()
    # Store the precise Unix timestamp
    node_packet["last_seen_timestamp"] = time.time()
    
    internal_db[data.room_id] = node_packet
    print(f"Update received from {data.room_id}")
    return {"status": "received"}

@app.get("/status")
async def get_status():
    current_time = time.time()
    results = []
    
    for room_id, data in internal_db.items():
        # Calculate time difference
        time_diff = current_time - data["last_seen_timestamp"]
        
        # Add a dynamic 'is_online' flag
        data["is_online"] = time_diff < 20
        results.append(data)
        
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)