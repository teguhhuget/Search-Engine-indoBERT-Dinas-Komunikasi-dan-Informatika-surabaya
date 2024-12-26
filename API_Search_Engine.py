from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from PRESENTASI_PKL import get_combined_ids  # Import the necessary function

app = FastAPI()

# Define the input model for the POST request
class SearchRequest(BaseModel):
    search: str

@app.post("/process")
async def process_query(request: SearchRequest):
    # Call the `get_combined_ids` function with the input query
    response_ids = get_combined_ids(request.search)  # Pass additional parameters if needed

    # Log the input and output in the terminal for verification
    print(f"Input query received: {request.search}")
    print(f"Output IDs: {response_ids}")

    # Check if the response contains results
    if response_ids == 0:
        return {"results": []}  # Return an empty list if no results are found
    else:
        return {"results": response_ids}  # Return all IDs as a list

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
