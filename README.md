## Install dependencies
```
pip install fastapi uvicorn torch torchvision scikit-learn joblib pillow
```

## Running the Server
```
uvicorn server:app --reload
```

Server runs at: `http://127.0.0.1:8000`

## Example request:
```bash
curl -X POST "http://127.0.0.1:8000/predict?clf=ann" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

## Example response:
```json
{
  "top": [
    {"index": 2, "label": "Moderate", "prob": 0.81},
    {"index": 1, "label": "Mild", "prob": 0.12}
  ],
  "raw": [0.01, 0.12, 0.81, 0.03, 0.03]
}
```
