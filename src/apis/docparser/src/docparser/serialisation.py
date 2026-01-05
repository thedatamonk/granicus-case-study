from pydantic import BaseModel

class IngestionResponse(BaseModel):
    response: str