from fastapi import FastAPI
import xgboost as xgb
import pandas as pd

app = FastAPI()

# Đánh thức "bộ não"
model = xgb.XGBClassifier()
model.load_model("xgboost_fire_model.json")

@app.get("/")
def home():
    return {"status": "AI RPG Tai Chinh dang hoat dong!"}

@app.get("/cham_diem")
def cham_diem(tien_mat: int, thu_nhap: int, chi_phi: int, tong_no: int):
    du_lieu = pd.DataFrame([[tien_mat, thu_nhap, chi_phi, tong_no]], 
                           columns=['tien_mat', 'thu_nhap', 'chi_phi', 'tong_no'])
    rui_ro = model.predict_proba(du_lieu)[0][1]
    return {"rui_ro": round(float(rui_ro) * 100, 2)}
