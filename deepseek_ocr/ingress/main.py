import os
import httpx
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse

app = FastAPI()

TARGET_URL = os.environ.get("TARGET_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY")

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy(request: Request, path_name: str):
    # Check API Key
    auth_header = request.headers.get("X-API-Key")
    if not auth_header or auth_header != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    url = f"{TARGET_URL}/{path_name}"
    
    # Filter headers to avoid conflicts
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("x-api-key", None)
    headers.pop("content-length", None)

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Build the request to the internal service
            proxy_req = client.build_request(
                request.method,
                url,
                headers=headers,
                content=request.stream(),
                params=request.query_params,
            )
            
            # Send request and stream response back
            proxy_resp = await client.send(proxy_req, stream=True)
            
            return StreamingResponse(
                proxy_resp.aiter_raw(),
                status_code=proxy_resp.status_code,
                headers=dict(proxy_resp.headers),
                background=None
            )
        except Exception as e:
             raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
