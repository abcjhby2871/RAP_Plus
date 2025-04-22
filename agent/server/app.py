# app.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from model_wrapper import LlavaAgent
from conversation_store import ConversationStore
import uvicorn
from pydantic import BaseModel
import shutil
import os

app = FastAPI()
conversation_store = ConversationStore()

# 初始化模型（只加载一次）


class DBUpdateRequest(BaseModel):
    database: str
    index_path: str | None = None

@app.post("/update_db")
def update_db(req: DBUpdateRequest):
    try:
        agent.update_database(req.database, req.index_path)
        return {"status": "success", "message": "Database updated successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/start")
def start_session():
    session_id = conversation_store.new_session(agent)
    return {"session_id": session_id}

@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    user_input: str = Form(...),
    image: UploadFile = None
):
    state = conversation_store.get(session_id)
    if state is None:
        return JSONResponse(status_code=400, content={"error": "Invalid session_id"})

    image_path = None
    if image:
        image_path = f"tmp_{image.filename}"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

    answer, updated_state = agent.chat(state, user_input, image_path)
    conversation_store.sessions[session_id] = updated_state

    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    return {"answer": answer}


if __name__ == "__main__":
    agent = LlavaAgent()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)