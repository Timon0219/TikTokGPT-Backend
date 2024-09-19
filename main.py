from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from query_service import query_sql, query_rag, hybrid_query
from pydantic import BaseModel

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:5173",  # Allow your frontend URL
    # Add other origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

class QueryRequest(BaseModel):
    query: str
    use_rag: bool

@app.post("/query")
async def handle_query(request: QueryRequest):
    print(request)
    user_query = request.query
    use_rag = request.use_rag

    if use_rag:
        result = query_rag(user_query)
    else:
        result = query_sql(user_query)

    return {"result": result}

@app.post("/hybrid")
async def handle_hybrid(request: QueryRequest):
    result = hybrid_query(request.query, request.use_rag)
    return {"result": result}
