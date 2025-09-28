"""
Main FastAPI application for handling travel agent queries.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from starlette.responses import JSONResponse
import os
import uuid
import datetime
from pydantic import BaseModel
from config.constant import logger
from agent.workflow import GraphBuilder
from utils.document import save_document
from dotenv import load_dotenv
load_dotenv()

""" Initialize FastAPI app """
app = FastAPI()

""" CORS middleware setup """
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

""" Pydantic model for chat request. """
class ChatRequest(BaseModel):
    message: str
    session_id : str = None  # Optional session ID
    


""" Health check endpoint """
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}

""" Endpoint to create a new session """
@app.post("/api/create_session")
async def create_session():
    try:
        logger.info("Creating new session...")
        session_id = str(uuid.uuid4())
        logger.info(f"New session created with ID: {session_id}")
        return {"session_id": session_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

""" Endpoint to save travel document """    
@app.post("/api/save_document")
async def save_travel_document(doc: dict):
    try:
        logger.info("Save Document Endpoint Hit")
        file_path = save_document(doc)
        return {"message": "Document saved successfully", "file_path": file_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

""" Endpoint to get session details (placeholder) """
@app.get("/api/get_session/{session_id}")
async def get_session(session_id: str):
    try:
        logger.info(f"Retrieving session with ID: {session_id}")
        # Placeholder for actual session retrieval logic
        return {"session_id": session_id, "status": "active"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

""" Endpoint to list available models """    
@app.get("/api/models")
async def list_models():
    try:
        logger.info("Listing available models...")
        # Placeholder for actual model listing logic
        models = ["openai", "gemini", "groq", "tavily"]
        return {"available_models": models}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

""" Endpoint to handle travel agent chat """
@app.post("/api/chat")
async def chat_travel_agent(chat:ChatRequest):
    try:
        logger.info("Travel Agent Chat Endpoint Hit")
        messages={"messages": [chat.message]}
        msg = f"Processing chat: {chat.message}"
        if chat.session_id is None:
            chat.session_id = str(uuid.uuid4()) # Generate new session ID
           
        logger.info("session_id: " + chat.session_id)

        graph = GraphBuilder(model_provider="openai")
        react_app=graph()
        #react_app = graph.build_graph()

        png_graph = react_app.get_graph().draw_mermaid_png()
        with open("my_graph.png", "wb") as f:
            f.write(png_graph)

        logger.info(f"Graph saved as 'my_graph.png' in {os.getcwd()}")  
        # Invoke the agent with the user's message
        
        output = react_app.invoke(messages)

        # If result is dict with messages:
        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content  # Last AI response
        else:
            final_output = str(output)
        
        return {"answer": final_output}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

""" Endpoint to handle weather-specific queries """
@app.post("/api/chat_weather")
async def chat_weather_agent(chat:ChatRequest):
    try:
        logger.info("Weather Agent Chat Endpoint Hit")
        messages={"messages": [chat.message]}
        msg = f"Processing chat: {chat.message}"
        if chat.session_id is None:
            chat.session_id = str(uuid.uuid4()) # Generate new session ID
           
        logger.info("session_id: " + chat.session_id)

        graph = GraphBuilder(model_provider="openai", use_weather_tool=True)
        react_app=graph()
        #react_app = graph.build_graph()

        png_graph = react_app.get_graph().draw_mermaid_png()
        with open("my_graph.png", "wb") as f:
            f.write(png_graph)

        logger.info(f"Graph saved as 'my_graph.png' in {os.getcwd()}")  
        # Invoke the agent with the user's message
        
        output = react_app.invoke(messages)

        # If result is dict with messages:
        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content  # Last AI response
        else:
            final_output = str(output)
        
        return {"answer": final_output}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get("/api/chat_api")
async def chat_api():
    try:
        logger.info("API Chat Endpoint Hit")
        return {"message": "API is working"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
        

""" Run the app with Uvicorn """
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Health Check at http://localhost:8000/api/health")    