from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.agent.workflow import GraphBuilder
from backend.utils.document import save_document
from starlette.responses import JSONResponse
import os
import uuid
import datetime
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from backend.config.constant import logger

""" Main FastAPI application for handling travel agent queries. """
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

""" Pydantic model for query request. """
class QueryRequest(BaseModel):
    message: str
    session_id : str = None  # Optional session ID

@app.get("/app/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}

@app.post("/app/create_session")
async def create_session():
    try:
        logger.info("Creating new session...")
        session_id = str(uuid.uuid4())
        logger.info(f"New session created with ID: {session_id}")
        return {"session_id": session_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/app/save_document")
async def save_travel_document(doc: dict):
    try:
        logger.info("Save Document Endpoint Hit")
        file_path = save_document(doc)
        return {"message": "Document saved successfully", "file_path": file_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/app/query")
async def query_travel_agent(query:QueryRequest):
    try:
        logger.info("Query Endpoint Hit")
        messages={"messages": [query.message]}
        msg = f"Processing query: {query.message}"
        if query.session_id is None:
            query.session_id = str(uuid.uuid4()) # Generate new session ID
           
        logger.info("session_id: " + query.session_id)

        graph = GraphBuilder(model_provider="groq")
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
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)    