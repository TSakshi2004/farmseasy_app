from flask import Flask, render_template, request, jsonify
import pickle, os, traceback

app = Flask(__name__)

MODELS_DIR = "models"

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Load models into memory if available
models = {}
for name in ["model_A.pkl", "model_B.pkl", "model_C.pkl"]:
    p = os.path.join(MODELS_DIR, name)
    if os.path.exists(p):
        try:
            models[name] = load_pkl(p)
        except Exception:
            # continue if pickle load fails for some model
            models[name] = None

def _try_transform(le, value):
    # helper: map value through label encoder, return transformed val or raise
    return le.transform([value])[0]

def predict_A(stage, region):
    m = models.get("model_A.pkl")
    if not m:
        return {"error": "Model A not trained yet."}
    stage = (stage or "").strip().lower()
    region = (region or "").strip().lower()
    try:
        s_enc = _try_transform(m["le_stage"], stage)
        r_enc = _try_transform(m["le_region"], region)
    except Exception:
        return {"error": "Unknown stage or region for Model A."}
    X = [[s_enc, r_enc]]
    try:
        pred = m["model"].predict(X)[0]
    except Exception:
        return {"error": "Model A prediction failed."}
    proba = None
    try:
        if hasattr(m["model"], "predict_proba"):
            proba = max(m["model"].predict_proba(X)[0])
    except Exception:
        proba = None
    label = m["le_disease"].inverse_transform([pred])[0]
    out = {"disease": label}
    if proba is not None:
        out["confidence"] = float(proba)
    return out

def predict_B(stage, region):
    m = models.get("model_B.pkl")
    if not m:
        return {"error": "Model B not trained yet."}
    stage = (stage or "").strip().lower()
    region = (region or "").strip().lower()
    try:
        s_enc = _try_transform(m["le_stage"], stage)
        r_enc = _try_transform(m["le_region"], region)
    except Exception:
        return {"error": "Unknown stage or region for Model B."}
    X = [[s_enc, r_enc]]
    try:
        pd_idx = m["clf_disease"].predict(X)[0]
        pc_idx = m["clf_cause"].predict(X)[0]
    except Exception:
        return {"error": "Model B prediction failed."}
    disease = m["le_disease"].inverse_transform([pd_idx])[0]
    cause = m["le_cause"].inverse_transform([pc_idx])[0]
    return {"disease": disease, "cause": cause}

def predict_C(stage, region, cause):
    m = models.get("model_C.pkl")
    if not m:
        return {"error": "Model C not trained yet."}
    stage = (stage or "").strip().lower()
    region = (region or "").strip().lower()
    cause = (cause or "").strip().lower()
    try:
        s_enc = _try_transform(m["le_stage"], stage)
        r_enc = _try_transform(m["le_region"], region)
        c_enc = _try_transform(m["le_cause"], cause)
    except Exception:
        return {"error": "Unknown stage/region/cause for Model C."}
    X = [[s_enc, r_enc, c_enc]]
    try:
        pred = m["model"].predict(X)[0]
    except Exception:
        return {"error": "Model C prediction failed."}
    proba = None
    try:
        if hasattr(m["model"], "predict_proba"):
            proba = max(m["model"].predict_proba(X)[0])
    except Exception:
        proba = None
    label = m["le_disease"].inverse_transform([pred])[0]
    out = {"disease": label}
    if proba is not None:
        out["confidence"] = float(proba)
    return out

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """AJAX JSON endpoint for frontend to call for predictions."""
    data = request.get_json(silent=True) or {}
    stage = (data.get("stage") or "").strip()
    region = (data.get("region") or "").strip()
    cause = (data.get("cause") or "").strip()
    want_b = data.get("want_b", False)
    try:
        if cause:
            result = predict_C(stage, region, cause)
        elif want_b:
            result = predict_B(stage, region)
        else:
            result = predict_A(stage, region)
    except Exception as e:
        result = {"error": f"Exception: {str(e)}"}
    return jsonify(result)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message = None
    # gather dropdown options from any available pickles
    stages = set()
    regions = set()
    causes = set()
    for key in ["model_A.pkl", "model_B.pkl", "model_C.pkl"]:
        p = os.path.join(MODELS_DIR, key)
        if os.path.exists(p):
            try:
                obj = load_pkl(p)
                if obj:
                    if "le_stage" in obj and getattr(obj["le_stage"], "classes_", None) is not None:
                        stages.update(list(obj["le_stage"].classes_))
                    if "le_region" in obj and getattr(obj["le_region"], "classes_", None) is not None:
                        regions.update(list(obj["le_region"].classes_))
                    if "le_cause" in obj and getattr(obj["le_cause"], "classes_", None) is not None:
                        causes.update(list(obj["le_cause"].classes_))
            except Exception:
                # ignore bad pickles when populating dropdowns
                continue
    stages = sorted(list(stages))
    regions = sorted(list(regions))
    causes = sorted(list(causes))

    if request.method == "POST":
        # keep server-side form handling fallback for non-js clients
        stage = request.form.get("stage", "").strip()
        region = request.form.get("region", "").strip()
        cause = request.form.get("cause", "").strip()
        want_b = request.form.get("want_b", "")  # checkbox to force model B
        try:
            if cause:
                result = predict_C(stage, region, cause)
            elif want_b:
                result = predict_B(stage, region)
            else:
                result = predict_A(stage, region)
        except Exception as e:
            message = "Error during prediction: " + str(e) + "\n" + traceback.format_exc()
    return render_template("index.html", stages=stages, regions=regions, causes=causes, result=result, message=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
