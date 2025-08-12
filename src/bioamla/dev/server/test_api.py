from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify/wav")
async def classify(file: UploadFile = File(...)):
  from bioamla.scripts.wav_ast_inference import wav_ast_inference
  import shutil
  from novus_pytils.files import delete_file
  try:
    file_path = f"{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = wav_ast_inference(file_path, "MIT/ast-finetuned-audioset-10-10-0.4593",16000)

    delete_file(file_path)

    return {"file":f"{file.filename}", "prediction":f"{prediction}"}
  except Exception as e:
    print(str(e))
    raise HTTPException(status_code=500, detail=str(e))

    
@app.post("/convert/wav")
async def convert_wav(file: UploadFile = File(...)):
  import shutil
  import os
  from novus_pytils.files import delete_file
  from bioamla.file.conversion import convert_wav 
  try:
    delete_file("out.mp3")
    file_path = f"{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    convert_wav(os.path.abspath(file_path), "mp3", ".mp3", new_file_name="out")
    #TODO this should delete this file but okay for now
    #TODO should write to a temp dir

    delete_file(file_path)
    filename=file_path.replace(".wav", ".mp3")
    return FileResponse(path="out.mp3", filename=filename)

  except Exception as e:
      print(str(e))
      raise HTTPException(status_code=500, detail=str(e))
  