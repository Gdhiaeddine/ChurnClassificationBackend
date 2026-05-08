from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from app.models import ChurnPredictionRequest, ChurnPredictionResponse
from app.predictor import predictor

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using XGBoost model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor.model is not None}


@app.post("/predict", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest):
    try:
        data = request.model_dump()
        prediction, probability = predictor.predict(data)
        churn_risk = predictor.get_churn_risk(probability)

        return ChurnPredictionResponse(
            prediction=prediction,
            probability=probability,
            churn_risk=churn_risk
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
