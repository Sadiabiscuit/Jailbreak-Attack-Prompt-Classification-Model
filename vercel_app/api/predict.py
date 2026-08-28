"""
Vercel serverless function — POST /api/predict
Loads the trained scikit-learn pipeline once, caches it across warm invocations,
and returns the predicted policy-violation class + probability distribution.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

# ---------------------------------------------------------------------------
# Make the deployment/ package importable so we can reuse clean_text() exactly
# as it was used during training.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEPLOYMENT_DIR = os.path.join(_PROJECT_ROOT, "deployment")
if _DEPLOYMENT_DIR not in sys.path:
    sys.path.insert(0, _DEPLOYMENT_DIR)

# ---------------------------------------------------------------------------
# Lazy-loaded, module-level cache — survives across warm invocations so we
# only deserialize the .joblib files on cold start.
# ---------------------------------------------------------------------------
_model = None
_vectorizer = None
_label_encoder = None
_clean_text = None
_load_error = None


def _load_artifacts():
    """Load model artifacts once and cache them at module scope."""
    global _model, _vectorizer, _label_encoder, _clean_text, _load_error

    if _model is not None:
        return  # already loaded

    try:
        import joblib
        from preprocessing import clean_text  # from deployment/

        _model = joblib.load(os.path.join(_DEPLOYMENT_DIR, "logistic_regression_model.joblib"))
        _vectorizer = joblib.load(os.path.join(_DEPLOYMENT_DIR, "tfidf_vectorizer.joblib"))
        _label_encoder = joblib.load(os.path.join(_DEPLOYMENT_DIR, "label_encoder.joblib"))
        _clean_text = clean_text
    except Exception as exc:
        _load_error = str(exc)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_INPUT_LENGTH = 5000


# ---------------------------------------------------------------------------
# Vercel handler (legacy Python runtime uses BaseHTTPRequestHandler)
# ---------------------------------------------------------------------------
class handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, body: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    # Handle CORS preflight
    def do_OPTIONS(self):
        self._send_json(200, {})

    def do_POST(self):
        # 1. Load artifacts (cached after first call)
        _load_artifacts()
        if _load_error is not None:
            self._send_json(500, {"error": f"Model failed to load: {_load_error}"})
            return

        # 2. Parse request body
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, ValueError):
            self._send_json(400, {"error": "Invalid JSON in request body."})
            return

        text = payload.get("text", "")

        # 3. Validate input
        if not isinstance(text, str) or text.strip() == "":
            self._send_json(400, {"error": "Text must be a non-empty string."})
            return

        if len(text) > MAX_INPUT_LENGTH:
            self._send_json(
                400,
                {"error": f"Input exceeds the {MAX_INPUT_LENGTH}-character limit."},
            )
            return

        # 4. Predict
        try:
            cleaned = _clean_text(text)
            X = _vectorizer.transform([cleaned])
            proba = _model.predict_proba(X)[0]

            predicted_index = proba.argmax()
            predicted_label = _label_encoder.inverse_transform([predicted_index])[0]
            confidence = float(proba[predicted_index])

            # Build full distribution sorted descending
            class_labels = _label_encoder.inverse_transform(range(len(proba)))
            sorted_indices = proba.argsort()[::-1]
            top_5 = [
                {"label": str(class_labels[i]), "probability": round(float(proba[i]), 6)}
                for i in sorted_indices[:5]
            ]

            self._send_json(200, {
                "prediction": str(predicted_label),
                "confidence": round(confidence, 6),
                "top_5": top_5,
            })
        except Exception as exc:
            self._send_json(500, {"error": f"Prediction failed: {str(exc)}"})
