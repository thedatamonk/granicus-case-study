from pydantic import BaseModel


class IngestionResponse(BaseModel):
    response: str

class HeartbeatResult(BaseModel):
    healthy: bool