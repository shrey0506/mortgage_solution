from fastapi import APIRouter
from mortgage.app.model.chat_model import ChatRequest, ChatResponse,InputData
from mortgage.app.controller.chat_controller import chatController,open_image
# from mortgage.src.utils.agents.agent import steer_mortgage_agent
from mortgage.app.controller.chat_controller import steer_rate_of_interest_controller
# from mortgage.src.utils.helper.ocr import text_extract
from fastapi import File, UploadFile, Form
from mortgage.src.ml_model_main import run_model
import base64
import pandas as pd

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    response = chatController(request.user_input)
    return ChatResponse(response=response)

@router.post("/aip", response_model=ChatResponse)
def aip_endpoint(request: ChatRequest):
    """Endpoint to Get AIP result."""
    response = steer_rate_of_interest_controller(request.user_input)
    return ChatResponse(response=str(response))

@router.post("/ocr", response_model=ChatResponse)
def ocr_endpoint(request: ChatRequest):
    """Endpoint to handle OCR requests."""
    # Placeholder for OCR processing logic
    response = text_extract(request.user_input, request.doc_type)
    return ChatResponse(response=response)


@router.post("/ocr", response_model=ChatResponse)
async def ocr_endpoint(
    app_id: str = Form(...),
    doc_type: str = Form(...),
    doc_name: str = Form(...),
    fileContent: UploadFile = File(...)
):
    UPLOAD_DIR = Path("mortgage/data")
    UPLOAD_DIR.mkdir(exist_ok=True)
    try:
        # Define save path
        file_path = UPLOAD_DIR / f"{doc_name}.png"
        
        # Save file to disk
        with open(file_path, "wb") as out_file:
            out_file.write(await fileContent.read())

        # Call OCR logic with file path
        response_text = text_extract(str(file_path), doc_type)

        return ChatResponse(response=response_text)

    except Exception as e:
        return ChatResponse(response=f"Error handling OCR request: {str(e)}")
    
    

@router.post("/pred")
def predict(input_data: InputData):
    input_df = pd.DataFrame([input_data.dict()])
    classification_pred, regression_pred = run_model(input_df)

    return {
        "mortgage_approved_prediction": classification_pred.tolist(),
        "interest_rate_prediction": regression_pred.tolist()
    }