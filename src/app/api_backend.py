import traceback
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

    ok = True
    msg = ""
    results = {}
    try:
        results = task.predict(input_data)
        msg = "success"
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception:
        ok = False
        msg = traceback.format_exc()

    # Return the processed data
    return {"ok": ok, "msg": msg, "results": results}


@app.get("/")
async def api():
    return FileResponse("./index.html", media_type="text/html")


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = (
        "%(asctime)s | " + log_config["formatters"]["access"]["fmt"]
    )
    log_config["formatters"]["default"]["fmt"] = (
        "%(asctime)s | " + log_config["formatters"]["default"]["fmt"]
    )
    uvicorn.run(
        "src.app.api_backend:app",
        host="0.0.0.0",
        port=7860,
        log_level="debug",
        log_config=log_config,
        reload=True,
    )
