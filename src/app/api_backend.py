from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rex.utils.initialization import set_seed_and_log_path

from src.task import SchemaGuidedInstructBertTask

set_seed_and_log_path(log_path="debug.log")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestData(BaseModel):
    data: List[Dict[str, Any]]


task = SchemaGuidedInstructBertTask.from_taskdir(
    "mirror_outputs/Mirror_Pretrain_AllExcluded_2",
    load_best_model=True,
    initialize=False,
    dump_configfile=False,
    update_config={
        "regenerate_cache": False,
    },
)


@app.post("/process")
def process_data(data: RequestData):
    input_data = data.data

    results = task.predict(input_data)

    # Return the processed data
    return {"results": results}


if __name__ == "__main__":
    uvicorn.run("src.app.api_backend:app", host="0.0.0.0", port=23333, reload=True)
