import json
import logging
import os
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flasgger import Swagger, swag_from
from flask import Flask, jsonify, make_response, render_template, request
from flask_caching import Cache
from flask_compress import Compress
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sklearn.calibration import CalibratedClassifierCV

from config import config
# config module is available if explicit app config is needed
from database import (Prediction, db, get_recent_predictions, get_statistics,
                      init_db, update_daily_stats)
from docs.swagger.config import swagger_config, swagger_template
from utils import MAX_REQUEST_TIMES, apply_defaults
from validators import LoanApplicationValidator

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Preserve insertion order of dicts when returning JSON (do not sort keys)
app.config["JSON_SORT_KEYS"] = False

# Load app configuration from config.py based on FLASK_ENV
env = os.environ.get("FLASK_ENV", "development")
app.config.from_object(config.get(env, config["default"]))

# Configure Flask extensions
cache = Cache(app)
# Normalize RATELIMIT_DEFAULT into a list of limit strings for Flask-Limiter
_rl = app.config.get("RATELIMIT_DEFAULT")
if isinstance(_rl, str):
    default_limits = [s.strip() for s in _rl.split(",") if s.strip()]
else:
    default_limits = _rl

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=default_limits,
    storage_uri=app.config.get("RATELIMIT_STORAGE_URL"),
)
compress = Compress(app)
# Enable CORS for browser-based clients (Swagger UI, Postman in browser, etc.)
# Allow all origins for documentation and development convenience; restrict in
# production if needed.
# Allow wildcard by default for development; in production set CORS_ORIGINS
# to a comma-separated list of allowed origins (e.g. https://yourdomain.com)
cors_origins = os.environ.get("CORS_ORIGINS", "*")
# Default as wildcard; may be a list of origins or the wildcard string
cors_arg: Union[str, List[str]] = "*"
if isinstance(cors_origins, str) and cors_origins.strip() != "*":
    cors_arg = [o.strip() for o in cors_origins.split(",") if o.strip()]

CORS(app, resources={r"/*": {"origins": cors_arg}}, supports_credentials=True)


# Ensure preflight requests are handled explicitly for strict browsers/clients
@app.before_request
def handle_options_preflight():
    # Short-circuit OPTIONS requests with appropriate CORS headers
    if request.method == "OPTIONS":
        resp = make_response("")
        origin = request.headers.get("Origin")
        if origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
        else:
            resp.headers["Access-Control-Allow-Origin"] = cors_arg
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers[
            "Access-Control-Allow-Headers"
        ] = "Origin, X-Requested-With, Content-Type, Accept, Authorization"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        return resp


# Initialize Swagger
try:
    swagger = Swagger(app, config=swagger_config, template=swagger_template)
    print("Swagger initialized successfully")
except Exception as e:
    print(f"Swagger initialization failed: {e}")
    import traceback

    traceback.print_exc()

# Initialize database
init_db(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Global variables
model = None
feature_names: List[str] = []
model_info: Dict[str, Any] = {}
validator = LoanApplicationValidator()

# Multi-model support
available_models = {}
model_comparison = {}
calibrated_models = {}


class CalibratedModelWrapper:
    """Wrapper that exposes a consistent classifier interface for a base model
    and a calibrator. The calibrator may be a CalibratedClassifierCV (accepts
    full feature X) or a Platt-scaling-like LogisticRegression trained on the
    base model's raw probability (1-D input). This wrapper delegates
    `predict` and `predict_proba` appropriately.
    """

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict(self, X):
        try:
            # Prefer calibrator's predict if available
            return self.calibrator.predict(X)
        except Exception:
            return self.base_model.predict(X)

    def predict_proba(self, X):
        # Try calibrator directly (works for CalibratedClassifierCV)
        try:
            return self.calibrator.predict_proba(X)
        except Exception:
            # Fallback: compute base raw proba and pass single-column to calibrator
            raw = self.base_model.predict_proba(X)[:, 1]
            return self.calibrator.predict_proba(raw.reshape(-1, 1))

    def __getattr__(self, name):
        # Delegate attribute access to base_model when not found on wrapper
        return getattr(self.base_model, name)


# Performance tracking
request_times = []


@app.before_request
def before_request():
    """Track request start time"""
    request.start_time = time.time()


@app.after_request
def after_request(response):
    """Track request duration and add headers"""
    if hasattr(request, "start_time"):
        duration = time.time() - request.start_time
        request_times.append(duration)

        # Keep only the most recent requests (bounded by configuration)
        if len(request_times) > MAX_REQUEST_TIMES:
            request_times.pop(0)

        # Add performance header
        response.headers["X-Response-Time"] = f"{duration:.3f}s"

    # Add cache headers. For frequently-changing analytics endpoints ensure
    # clients do not reuse stale browser cache. Other GETs may be cached for
    # short durations to improve performance.
    if request.method == "GET":
        # Endpoints that should always be fetched fresh by clients
        no_cache_paths = ("/statistics", "/analytics", "/history")
        if any(request.path.startswith(p) for p in no_cache_paths):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        else:
            response.headers["Cache-Control"] = "public, max-age=300"

    return response


def load_all_models():
    """Load all available models"""
    # model_comparison is rebound below; declare global to update module var
    global model_comparison

    model_files = {
        "gradient_boosting": "models/gradient_boosting.pkl",
        "random_forest": "models/random_forest.pkl",
        "logistic_regression": "models/logistic_regression.pkl",
        "support_vector_machine": "models/support_vector_machine.pkl",
    }

    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            try:
                available_models[model_name] = joblib.load(model_path)
                logger.info("[OK] Loaded %s", model_name)
                # Diagnostic: log model type and expected features when available
                loaded = available_models[model_name]
                try:
                    is_pipeline = hasattr(loaded, "steps") and isinstance(
                        getattr(loaded, "steps"), list
                    )
                    fnames = list(getattr(loaded, "feature_names_in_", []))
                    logger.info(
                        "[DEBUG] Model=%s pipeline=%s feature_count=%d",
                        model_name,
                        is_pipeline,
                        len(fnames),
                    )
                    if fnames:
                        logger.info(
                            "[DEBUG] %s feature_names_in_: %s", model_name, fnames
                        )
                except Exception:
                    logger.debug("[DEBUG] Could not introspect model %s", model_name)
            except Exception as e:
                logger.warning(f"[WARN] Could not load {model_name}: {str(e)}")

    # Load comparison data
    try:
        with open("models/model_comparison.json", "r") as f:
            model_comparison = json.load(f)
        logger.info("[OK] Loaded model comparison data")
    except Exception as e:
        logger.warning(f"[WARN] Could not load comparison data: {str(e)}")

    logger.info("[OK] Total models loaded: %d", len(available_models))
    # Attempt to load persisted calibrators and wrap them with the base model
    try:
        calibrator_dir = os.path.join("models", "calibrators")
        if os.path.isdir(calibrator_dir):
            for fname in os.listdir(calibrator_dir):
                if fname.endswith(".pkl"):
                    key = fname[:-4]
                    path = os.path.join(calibrator_dir, fname)
                    try:
                        loaded_cal = joblib.load(path)
                        # If we have a base model loaded, wrap the calibrator so
                        # callers can safely call predict/predict_proba on it.
                        if key in available_models:
                            try:
                                calibrated_models[key] = CalibratedModelWrapper(
                                    available_models[key], loaded_cal
                                )
                                logger.info(
                                    "[CAL] Loaded persisted calibrator for %s from %s",
                                    key,
                                    path,
                                )
                            except Exception:
                                # Fallback: store raw calibrator
                                calibrated_models[key] = loaded_cal
                                logger.debug(
                                    "[CAL] Loaded calibrator for %s; raw fallback",
                                    key,
                                )
                        else:
                            calibrated_models[key] = loaded_cal
                            logger.info(
                                "[CAL] Loaded calibrator for %s (no base model)",
                                key,
                            )
                    except Exception:
                        logger.debug("[CAL] Could not load calibrator %s", path)
    except Exception:
        logger.debug("[CAL] No persisted calibrators found")
    return len(available_models) > 0


def calibrate_models(calibration_csv_path="data/train_u6lujuX_CVtuZ9i.csv"):
    """
    Calibrate loaded models using a held-out portion of training data.
    This uses `CalibratedClassifierCV` with `cv='prefit'` so it requires
    the models to be already trained and loaded in `available_models`.
    """
    # Mutates module-level dicts; no global binding required

    if not available_models:
        logger.info("[CAL] No available models to calibrate")
        return

    # Try to load calibration data
    try:
        df = pd.read_csv(calibration_csv_path)
        logger.info(
            "[CAL] Loaded calibration data from %s: %d rows",
            calibration_csv_path,
            len(df),
        )
    except Exception as e:
        logger.warning("[CAL] Could not load calibration CSV: %s", e)
        return

    # Prepare labels (Loan_Status -> 1/0)
    if "Loan_Status" not in df.columns:
        logger.warning("[CAL] Calibration data missing Loan_Status column")
        return
    y = df["Loan_Status"].map({"Y": 1, "N": 0}).fillna(0).astype(int)

    # Build full engineered DataFrame similar to preprocess_input but vectorized
    # Ensure numeric conversions
    df["ApplicantIncome"] = pd.to_numeric(
        df["ApplicantIncome"], errors="coerce"
    ).fillna(0)
    df["CoapplicantIncome"] = pd.to_numeric(
        df.get("CoapplicantIncome", 0), errors="coerce"
    ).fillna(0)
    df["LoanAmount"] = pd.to_numeric(df.get("LoanAmount", 0), errors="coerce").fillna(0)
    df["Loan_Amount_Term"] = pd.to_numeric(
        df.get("Loan_Amount_Term", 360), errors="coerce"
    ).fillna(360)
    df["Credit_History"] = pd.to_numeric(
        df.get("Credit_History", 1), errors="coerce"
    ).fillna(1)

    # Fill categorical defaults
    df["Gender"] = df["Gender"].fillna("Male")
    df["Married"] = df["Married"].fillna("Yes")
    df["Dependents"] = df["Dependents"].fillna("0").replace("3+", "3")
    df["Education"] = df["Education"].fillna("Graduate")
    df["Self_Employed"] = df["Self_Employed"].fillna("No")
    df["Property_Area"] = df["Property_Area"].fillna("Urban")

    # Engineered features
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Income_Loan_Ratio"] = df["TotalIncome"] / (df["LoanAmount"] + 1)
    df["Loan_Amount_Per_Term"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] / 12 + 1)
    df["Balance_Income"] = df["TotalIncome"] - (df["EMI"] * 1000)
    df["Log_ApplicantIncome"] = np.log1p(df["ApplicantIncome"])
    df["Log_CoapplicantIncome"] = np.log1p(df["CoapplicantIncome"])
    df["Log_LoanAmount"] = np.log1p(df["LoanAmount"])
    df["Log_TotalIncome"] = np.log1p(df["TotalIncome"])

    # Encoded columns
    gender_map = {"Male": 1, "Female": 0}
    df["Gender_Encoded"] = df["Gender"].map(gender_map).fillna(1)
    married_map = {"Yes": 1, "No": 0}
    df["Married_Encoded"] = df["Married"].map(married_map).fillna(1)
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0)
    df["Dependents_Encoded"] = df["Dependents"]
    education_map = {"Graduate": 1, "Not Graduate": 0}
    df["Education_Encoded"] = df["Education"].map(education_map).fillna(1)
    self_employed_map = {"Yes": 1, "No": 0}
    df["Self_Employed_Encoded"] = df["Self_Employed"].map(self_employed_map).fillna(0)
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    df["Property_Area_Encoded"] = df["Property_Area"].map(property_map).fillna(2)

    # Prepare raw_fields used by pipelines
    raw_fields = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    # Calibrate each model where possible
    for model_name, clf in list(available_models.items()):
        try:
            # Determine expected features for this model
            fnames = getattr(clf, "feature_names_in_", None)
            model_is_pipeline = hasattr(clf, "steps") and isinstance(
                getattr(clf, "steps"), list
            )

            if model_is_pipeline:
                X = df[raw_fields].copy()
            else:
                expected = (
                    list(fnames)
                    if fnames is not None and len(fnames) > 0
                    else feature_names
                )
                X = build_model_input(df, expected)

            # Only calibrate if classifier supports predict_proba
            if hasattr(clf, "predict_proba"):
                # Try CalibratedClassifierCV when available (newer sklearn), otherwise
                # fall back to a Platt-scaling style LogisticRegression on the raw probs.
                try:
                    try:
                        calibrator = CalibratedClassifierCV(estimator=clf, cv="prefit")
                    except TypeError:
                        calibrator = CalibratedClassifierCV(
                            base_estimator=clf, cv="prefit"
                        )

                    calibrator.fit(X, y)
                    # Wrap calibrator so it behaves like a full model
                    try:
                        calibrated_models[model_name] = CalibratedModelWrapper(
                            clf, calibrator
                        )
                    except Exception:
                        calibrated_models[model_name] = calibrator
                    logger.info(
                        "[CAL] Calibrated model: %s (CalibratedClassifierCV)",
                        model_name,
                    )
                    # Persist calibrator for reuse across restarts
                    try:
                        calibrator_dir = os.path.join("models", "calibrators")
                        os.makedirs(calibrator_dir, exist_ok=True)
                        path = os.path.join(calibrator_dir, f"{model_name}.pkl")
                        joblib.dump(calibrated_models[model_name], path)
                        logger.info(
                            "[CAL] Persisted calibrator for %s to %s",
                            model_name,
                            path,
                        )
                    except Exception:
                        logger.warning(
                            "[CAL] Failed to persist calibrator for %s",
                            model_name,
                        )
                except Exception:
                    # Fallback: train a LogisticRegression on predicted
                    # probabilities (Platt scaling) using a dynamic import to avoid
                    # static analyzer errors resolving sklearn.linear_model.
                    try:
                        import importlib

                        # Try to load the public module first, then an internal module
                        try:
                            lr_mod = importlib.import_module("sklearn.linear_model")
                        except Exception:
                            lr_mod = importlib.import_module(
                                "sklearn.linear_model._logistic"
                            )

                        LogisticRegression = getattr(lr_mod, "LogisticRegression")

                        raw_proba = clf.predict_proba(X)[:, 1]
                        platt = LogisticRegression(solver="lbfgs")
                        platt.fit(raw_proba.reshape(-1, 1), y)
                        # Store platt calibrator wrapped so callers can safely use it
                        try:
                            calibrated_models[model_name] = CalibratedModelWrapper(
                                clf, platt
                            )
                        except Exception:
                            calibrated_models[model_name] = platt
                        logger.info(
                            "[CAL] Calibrated model: %s (Platt scaling)", model_name
                        )
                        # Persist platt calibrator
                        try:
                            calibrator_dir = os.path.join("models", "calibrators")
                            os.makedirs(calibrator_dir, exist_ok=True)
                            path = os.path.join(calibrator_dir, f"{model_name}.pkl")
                            joblib.dump(calibrated_models[model_name], path)
                            logger.info(
                                "[CAL] Persisted calibrator for %s to %s",
                                model_name,
                                path,
                            )
                        except Exception:
                            logger.warning(
                                "[CAL] Failed to persist platt calibrator for %s",
                                model_name,
                            )
                    except Exception as e:
                        logger.warning(
                            "[CAL] Calibration failed for %s: %s", model_name, e
                        )
                else:
                    logger.info(
                        "[CAL] Skipping calibration for %s (no predict_proba)",
                        model_name,
                    )
        except Exception as e:
            logger.warning(
                "[CAL] Calibration failed for %s: %s",
                model_name,
                e,
            )

    # Optionally, compute ensemble weights from model_comparison (accuracy)
    try:
        weights = {}
        total = 0.0
        for k, v in model_comparison.get("models", {}).items():
            name_key = k.lower().replace(" ", "_")
            acc = float(v.get("accuracy", 0.0))
            weights[name_key] = acc
            total += acc
        if total > 0:
            # normalize
            for k in list(weights.keys()):
                weights[k] = weights[k] / total
        model_comparison["ensemble_weights"] = weights
        logger.info("[CAL] Ensemble weights added to model_comparison")
    except Exception:
        logger.debug("[CAL] Could not compute ensemble weights")
    # Persist updated model_comparison to disk so ensemble weights survive restarts
    try:
        comp_path = os.path.join("models", "model_comparison.json")
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(model_comparison, f, indent=2)
        logger.info("[CAL] Persisted updated model_comparison to %s", comp_path)
    except Exception:
        logger.debug("[CAL] Failed to persist model_comparison.json")


def compute_ensemble_prob(full_df):
    """
    Compute a weighted ensemble probability (approved) across available models.
    Returns a dict with 'approved' and 'rejected' probabilities and per-model probs.
    """
    probs = {}
    weights = model_comparison.get("ensemble_weights", {})

    # Raw fields used by pipeline models
    raw_fields = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    total_weight = 0.0
    agg_approved = 0.0
    agg_rejected = 0.0

    for model_name, clf in available_models.items():
        try:
            model_used = calibrated_models.get(model_name, clf)

            model_is_pipeline = hasattr(clf, "steps") and isinstance(
                getattr(clf, "steps"), list
            )
            fnames = getattr(clf, "feature_names_in_", None)

            if model_is_pipeline:
                X = full_df[raw_fields].copy()
            else:
                expected = (
                    list(fnames)
                    if fnames is not None and len(fnames) > 0
                    else feature_names
                )
                X = build_model_input(full_df, expected)

            # Ensure the model supports probability estimates
            if not hasattr(model_used, "predict_proba"):
                continue

            # If we have a persisted/created calibrator, it might be one of:
            # - a CalibratedClassifierCV (accepts X as original features)
            # - a Platt-scaling LogisticRegression trained on raw probs (accepts 1-column input)
            if model_name in calibrated_models:
                calib = calibrated_models[model_name]
                # Try direct predict_proba with the calibrator (works for CalibratedClassifierCV)
                try:
                    # Diagnostic: log dtypes and sample values to help debug conversion errors
                    try:
                        logger.info(
                            f"[CAL_DIAG] Calibrator type={type(calib)} for model={model_name}"
                        )
                        logger.info(
                            f"[CAL_DIAG] X dtypes: {dict(X.dtypes.apply(lambda x: x.name))}"
                        )
                        logger.info(f"[CAL_DIAG] X sample: {X.iloc[0].to_dict()}")
                    except Exception:
                        logger.debug("[CAL_DIAG] Could not serialize X diagnostics")

                    proba = calib.predict_proba(X)[0]
                except Exception as e:
                    logger.warning(
                        f"[CAL_DIAG] calib.predict_proba failed for {model_name}: {e}"
                    )
                    # Fallback: compute raw model proba then pass single-column to calibrator
                    try:
                        raw_proba = clf.predict_proba(X)[:, 1]
                    except Exception as e2:
                        logger.error(
                            f"[CAL_DIAG] raw model predict_proba failed for {model_name}: {e2}",
                            exc_info=True,
                        )
                        raise
                    proba = calib.predict_proba(raw_proba.reshape(-1, 1))[0]
            else:
                proba = model_used.predict_proba(X)[0]

            approved_p = float(proba[1])
            rejected_p = float(proba[0])

            # weight by ensemble_weights if available, else fall back to model_comparison accuracy
            w = weights.get(model_name, None)
            if w is None:
                mc = model_comparison.get("models", {}).get(
                    model_name.replace("_", " ").title(), {}
                )
                try:
                    w = float(mc.get("accuracy", 0.0))
                except Exception:
                    w = 0.0

            probs[model_name] = {
                "approved": approved_p,
                "rejected": rejected_p,
                "weight": w,
            }
            agg_approved += approved_p * w
            agg_rejected += rejected_p * w
            total_weight += w
        except Exception:
            continue

    if total_weight > 0:
        agg_approved /= total_weight
        agg_rejected /= total_weight
    else:
        # default to 0.5 each
        agg_approved = 0.5
        agg_rejected = 0.5

    return {"approved": agg_approved, "rejected": agg_rejected, "per_model": probs}


def preprocess_input(data: dict, select_features: bool = True) -> pd.DataFrame:
    """
    Preprocess input data to match training data format.

    Handles missing values, performs feature engineering, and encodes
    categorical variables to prepare data for model prediction.

    Args:
        data (dict): Raw input data from API request containing loan
                     application information
        select_features (bool): If True, return only selected features.

    Returns:
        pd.DataFrame: Preprocessed data with required features for prediction

    Example:
        >>> data = {'ApplicantIncome': 5000, 'LoanAmount': 150}
        >>> df = preprocess_input(data)
        >>> df.shape
        (1, 15)
    """

    # Apply default values for missing keys first
    data = apply_defaults(data)

    # Create DataFrame from input
    df = pd.DataFrame([data])

    # Handle missing values (use same logic as training)
    # Numeric columns - fill with median (use training medians)
    # Ensure ApplicantIncome exists (benchmarks may send empty payload)
    # Numeric fallbacks (apply_defaults already handled common defaults)
    if "ApplicantIncome" not in df.columns or pd.isna(df["ApplicantIncome"].iloc[0]):
        df["ApplicantIncome"] = 0
    if "LoanAmount" not in df.columns or pd.isna(df["LoanAmount"].iloc[0]):
        df["LoanAmount"] = 128.0

    if "Loan_Amount_Term" not in df.columns or pd.isna(df["Loan_Amount_Term"].iloc[0]):
        df["Loan_Amount_Term"] = 360.0

    if "Credit_History" not in df.columns or pd.isna(df["Credit_History"].iloc[0]):
        df["Credit_History"] = 1.0

    # CoapplicantIncome - default to 0
    if "CoapplicantIncome" not in df.columns:
        df["CoapplicantIncome"] = 0

    # Categorical defaults
    if "Gender" not in df.columns:
        df["Gender"] = "Male"
    if "Married" not in df.columns:
        df["Married"] = "Yes"
    if "Dependents" not in df.columns:
        df["Dependents"] = "0"
    if "Education" not in df.columns:
        df["Education"] = "Graduate"
    if "Self_Employed" not in df.columns:
        df["Self_Employed"] = "No"
    if "Property_Area" not in df.columns:
        df["Property_Area"] = "Urban"

    # ============================================================================
    # FEATURE ENGINEERING (MUST MATCH TRAINING)
    # ============================================================================

    # Total Income (note: use underscore for model compatibility)
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["TotalIncome"] = df[
        "Total_Income"
    ]  # Also create non-underscore version for compatibility

    # Loan to Income Ratio (LoanAmount / TotalIncome)
    df["Loan_to_Income_Ratio"] = df["Total_Income"] / (df["Total_Income"] + 1)
    # Prevent division by zero - if income is 0, ratio is 1
    df["Loan_to_Income_Ratio"] = df["LoanAmount"] / (df["Total_Income"] + 1)

    # Income to Loan Ratio (for compatibility)
    df["Income_Loan_Ratio"] = df["Total_Income"] / (df["LoanAmount"] + 1)

    # Loan Amount per Term
    df["Loan_Amount_Per_Term"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)

    # Income per Dependent
    dependents_val = (
        pd.to_numeric(df["Dependents"], errors="coerce").fillna(0).astype(int)
    )
    df["Income_per_Dependent"] = df["Total_Income"] / (dependents_val.replace(0, 1))

    # EMI (Estimated Monthly Installment)
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] / 12 + 1)

    # Balance Income (Income after EMI)
    df["Balance_Income"] = df["Total_Income"] - (df["EMI"] * 1000)

    # Log transformations for skewed features
    df["Log_ApplicantIncome"] = np.log1p(df["ApplicantIncome"])
    df["Log_CoapplicantIncome"] = np.log1p(df["CoapplicantIncome"])
    df["Log_LoanAmount"] = np.log1p(df["LoanAmount"])
    df["Log_TotalIncome"] = np.log1p(df["Total_Income"])

    # ============================================================================
    # ENCODE CATEGORICAL VARIABLES
    # ============================================================================

    # Map Gender
    gender_map = {"Male": 1, "Female": 0}
    df["Gender_Encoded"] = df["Gender"].map(gender_map).fillna(1)

    # Map Married
    married_map = {"Yes": 1, "No": 0}
    df["Married_Encoded"] = df["Married"].map(married_map).fillna(1)

    # Map Dependents
    df["Dependents"] = df["Dependents"].replace("3+", "3")
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0)
    df["Dependents_Encoded"] = df["Dependents"]

    # Map Education
    education_map = {"Graduate": 1, "Not Graduate": 0}
    df["Education_Encoded"] = df["Education"].map(education_map).fillna(1)

    # Map Self_Employed
    self_employed_map = {"Yes": 1, "No": 0}
    df["Self_Employed_Encoded"] = df["Self_Employed"].map(self_employed_map).fillna(0)

    # One-hot encode Property_Area (create Property_Semiurban and Property_Urban)
    df["Property_Semiurban"] = (df["Property_Area"] == "Semiurban").astype(int)
    df["Property_Urban"] = (df["Property_Area"] == "Urban").astype(int)

    # Map encoded columns back to original names when models expect those names
    # This ensures we don't pass raw string categorical columns like 'Male' to models
    try:
        if "Gender_Encoded" in df.columns:
            df["Gender"] = df["Gender_Encoded"]
        if "Married_Encoded" in df.columns:
            df["Married"] = df["Married_Encoded"]
        if "Education_Encoded" in df.columns:
            df["Education"] = df["Education_Encoded"]
        if "Self_Employed_Encoded" in df.columns:
            df["Self_Employed"] = df["Self_Employed_Encoded"]
        if "Dependents_Encoded" in df.columns:
            df["Dependents"] = df["Dependents_Encoded"]
    except Exception:
        logger.debug("[DIAG] Could not map encoded columns back to original names")

    # ============================================================================
    # SELECT ONLY REQUIRED FEATURES IN CORRECT ORDER
    # ============================================================================

    # At this point `df` contains both raw and engineered/encoded features.
    if select_features:
        # Ensure all required features exist and return in correct order
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0

        df = df[feature_names]
        return df

    # If not selecting, return the full engineered DataFrame
    # so callers can pick model-specific features
    return df


def build_model_input(full_df, expected_features):
    """
    Build a DataFrame for a specific model given the full engineered dataframe.
    This attempts to map or compute columns so the returned DataFrame has
    exactly the `expected_features` (names and order) that the model was
    trained with.

    Parameters:
    - full_df: DataFrame returned by `preprocess_input(select_features=False)`
    - expected_features: list of feature names expected by the model

    Returns: DataFrame with columns in `expected_features` order
    """
    df = full_df.copy()

    # Helpers for derived values
    def ensure_total_income(d):
        if "TotalIncome" in d.columns:
            return d["TotalIncome"]
        return d["ApplicantIncome"] + d["CoapplicantIncome"]

    def ensure_dependents(d):
        if "Dependents" in d.columns:
            return d["Dependents"]
        if "Dependents_Encoded" in d.columns:
            return d["Dependents_Encoded"]
        return pd.Series(0, index=d.index)

    # Ensure some commonly needed intermediate values exist
    if "TotalIncome" not in df.columns:
        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    if (
        "Income_Loan_Ratio" not in df.columns
        and "LoanAmount" in df.columns
        and "TotalIncome" in df.columns
    ):
        df["Income_Loan_Ratio"] = df["TotalIncome"] / (df["LoanAmount"] + 1)
    if (
        "Loan_to_Income_Ratio" not in df.columns
        and "LoanAmount" in df.columns
        and "TotalIncome" in df.columns
    ):
        # Loan_to_Income_Ratio = LoanAmount / TotalIncome (avoid div0)
        df["Loan_to_Income_Ratio"] = df["LoanAmount"] / (df["TotalIncome"] + 1)
    if "Income_per_Dependent" not in df.columns:
        deps = ensure_dependents(df)
        df["Income_per_Dependent"] = df["TotalIncome"] / (deps.replace(0, 1))

    # Create one-hot style property area columns if not present
    if "Property_Semiurban" not in df.columns and "Property_Urban" not in df.columns:
        if "Property_Area" in df.columns:
            df["Property_Semiurban"] = (df["Property_Area"] == "Semiurban").astype(int)
            df["Property_Urban"] = (df["Property_Area"] == "Urban").astype(int)
        elif "Property_Area_Encoded" in df.columns:
            df["Property_Semiurban"] = (df["Property_Area_Encoded"] == 1).astype(int)
            df["Property_Urban"] = (df["Property_Area_Encoded"] == 2).astype(int)
        else:
            df["Property_Semiurban"] = 0
            df["Property_Urban"] = 0

    # Map encoded columns back to original names when model expects raw names
    # e.g., model trained with column named 'Gender' but current df has 'Gender_Encoded'
    mapping_candidates = {
        "Gender": ["Gender", "Gender_Encoded"],
        "Married": ["Married", "Married_Encoded"],
        "Dependents": ["Dependents", "Dependents_Encoded"],
        "Education": ["Education", "Education_Encoded"],
        "Self_Employed": ["Self_Employed", "Self_Employed_Encoded"],
        "Total_Income": ["Total_Income", "TotalIncome"],
        "Loan_to_Income_Ratio": ["Loan_to_Income_Ratio", "Income_Loan_Ratio"],
        "Income_per_Dependent": [
            "Income_per_Dependent",
            "Income_per_Dependent",
            "Income_per_Dependent",
        ],
        "Balance_Income": ["Balance_Income"],
        "EMI": ["EMI"],
        "Log_ApplicantIncome": ["Log_ApplicantIncome"],
        "Log_CoapplicantIncome": ["Log_CoapplicantIncome"],
        "Log_LoanAmount": ["Log_LoanAmount"],
        "Log_TotalIncome": ["Log_TotalIncome"],
    }

    result_cols = {}
    sources = {}
    for feat in expected_features:
        # Prefer numeric/encoded versions when available
        # (avoid passing strings like 'Male' to models)
        enc = f"{feat}_Encoded"

        # If feature column exists directly, prefer a numeric representation when possible
        if feat in df.columns:
            col = df[feat]
            if (
                (pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col))
                and enc in df.columns
                and pd.api.types.is_numeric_dtype(df[enc])
            ):
                result_cols[feat] = df[enc]
                sources[feat] = enc
                continue
            if pd.api.types.is_numeric_dtype(col):
                result_cols[feat] = col
                sources[feat] = feat
                continue
            # fallback to the column as-is
            result_cols[feat] = col
            sources[feat] = feat
            continue

        # Try candidate mappings, preferring encoded/numeric candidates first
        used = False
        if feat in mapping_candidates:
            candidates = mapping_candidates[feat]
            # sort so that encoded-like candidates come first
            candidates_sorted = sorted(
                candidates, key=lambda x: (0 if x.endswith("_Encoded") else 1, x)
            )
            for cand in candidates_sorted:
                if cand in df.columns:
                    result_cols[feat] = df[cand]
                    sources[feat] = cand
                    used = True
                    break

        if used:
            continue

        # Special-case computations
        if (
            feat == "Loan_to_Income_Ratio"
            and "LoanAmount" in df.columns
            and "TotalIncome" in df.columns
        ):
            result_cols[feat] = df["LoanAmount"] / (df["TotalIncome"] + 1)
            sources[feat] = "computed:Loan_to_Income_Ratio"
            continue

        if feat == "Income_per_Dependent" and "TotalIncome" in df.columns:
            deps = ensure_dependents(df)
            result_cols[feat] = df["TotalIncome"] / (deps.replace(0, 1))
            sources[feat] = "computed:Income_per_Dependent"
            continue

        if feat in ("Property_Semiurban", "Property_Urban") and feat not in df.columns:
            # Already ensured these exist above, but defensive fallback
            if feat == "Property_Semiurban":
                result_cols[feat] = (
                    (df.get("Property_Area") == "Semiurban").astype(int)
                    if "Property_Area" in df.columns
                    else df.get("Property_Semiurban", 0)
                )
                sources[feat] = "computed:Property_Semiurban"
            else:
                result_cols[feat] = (
                    (df.get("Property_Area") == "Urban").astype(int)
                    if "Property_Area" in df.columns
                    else df.get("Property_Urban", 0)
                )
                sources[feat] = "computed:Property_Urban"
            continue

        # As a last resort, if there's an encoded version of the same logical name, try that
        if enc in df.columns:
            result_cols[feat] = df[enc]
            sources[feat] = enc
            continue

        # Give up: fill with zeros to keep shape
        result_cols[feat] = pd.Series(0, index=df.index)
        sources[feat] = "filled_zero"

    # Build DataFrame in requested order
    model_df = pd.DataFrame(result_cols)
    model_df = model_df[expected_features]
    try:
        logger.info(f"[DIAG] build_model_input sources: {sources}")
    except Exception:
        logger.debug("[DIAG] build_model_input completed (could not serialize sources)")
    return model_df


@app.route("/")
@cache.cached(timeout=60)
def home():
    """API information endpoint"""
    # Return API information as JSON. Do not redirect browsers to /app
    # (startup still opens /app via webbrowser.open in __main__).
    stats = get_statistics()

    logger.debug(f"[DEBUG] model_info keys order: {list(model_info.keys())}")
    return jsonify(
        {
            "name": "Loan Prediction API",
            "version": "7.0",
            "status": "running",
            "model_loaded": model is not None,
            "models_available": len(available_models),
            "model_accuracy": f"{model_info.get('accuracy', 0):.2%}",
            "database": "connected",
            "documentation": "/docs",
            "statistics": stats,
            "features": {
                "validation": "Comprehensive input validation",
                "database": "Prediction history storage",
                "analytics": "Usage statistics and trends",
                "documentation": "Interactive Swagger UI",
                "caching": "Response caching for performance",
                "rate_limiting": "API rate limiting",
                "compression": "Response compression",
                "multi_model": "Multiple ML models available",
            },
            "endpoints": {
                "/": "API information",
                "/docs": "Interactive API documentation",
                "/health": "Health check",
                "/predict": "Make loan prediction (POST)",
                "/predict/<model>": "Predict with specific model",
                "/models": "List available models",
                "/models/compare": "Compare model performance",
                "/models/benchmark": "Benchmark all models (POST)",
                "/history": "Recent predictions",
                "/statistics": "Overall statistics",
                "/performance": "Performance metrics",
                "/app": "Frontend application",
            },
        }
    )


@app.route("/app")
def frontend():
    """Serve frontend application"""
    # Reload model comparison so UI reflects any recent training/tuning results
    try:
        comp_path = os.path.join("models", "model_comparison.json")
        if os.path.exists(comp_path):
            with open(comp_path, "r", encoding="utf-8") as f:
                comp = json.load(f)
            # Try to populate a fresh model_info for the default model
            comp_key = "Random Forest"
            info = comp.get("models", {}).get(comp_key, {})
            local_model_info = {
                "name": "random_forest",
                "accuracy": info.get("accuracy"),
                "precision": info.get("precision"),
                "recall": info.get("recall"),
                "f1_score": info.get("f1_score"),
            }
            return render_template("index.html", model_info=local_model_info)
    except Exception:
        # Fallback to previously loaded model_info
        pass

    # Pass model_info to template so the frontend can display dynamic metadata
    return render_template("index.html", model_info=model_info)


@app.route("/health")
@limiter.exempt
@swag_from(os.path.join(os.path.dirname(__file__), "docs", "swagger", "health.yml"))
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "models_available": len(available_models),
            "database": "connected",
            "cache": "enabled",
            "compression": "enabled",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/predict", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "docs", "swagger", "predict.yml"))
@limiter.limit("100 per hour")
def predict():
    """Predict loan approval and store in database"""

    try:
        logger.info(f"Prediction request from {request.remote_addr}")

        if not model:
            logger.error("Model not loaded")
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        if not data:
            return (
                jsonify(
                    {
                        "error": "No data provided",
                        "message": "Please send JSON data in request body",
                        "required_fields": ["ApplicantIncome"],
                        "documentation": "/docs",
                    }
                ),
                400,
            )

        # Validate input
        is_valid, errors, warnings = validator.validate_loan_application(data)

        if not is_valid:
            logger.warning(f"Validation failed: {errors}")
            return (
                jsonify(
                    {
                        "error": "Validation failed",
                        "validation_errors": errors,
                        "required_fields": ["ApplicantIncome"],
                        "documentation": "/docs",
                    }
                ),
                400,
            )

        if warnings:
            logger.info(f"Validation warnings: {warnings}")

        # Preprocess and predict
        df_processed = preprocess_input(data)
        prediction = model.predict(df_processed)[0]
        prediction_proba = model.predict_proba(df_processed)[0]

        # Prepare result
        result = {
            "success": True,
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "prediction_code": int(prediction),
            "confidence": float(max(prediction_proba)),
            "probability": {
                "rejected": float(prediction_proba[0]),
                "approved": float(prediction_proba[1]),
            },
            "input_data": data,
            "model_info": {"accuracy": model_info.get("accuracy"), "version": "2.0"},
            "timestamp": datetime.now().isoformat(),
        }

        if warnings:
            result["warnings"] = warnings

        # Add ensemble result (if calibrated models exist)
        try:
            full_df = preprocess_input(data, select_features=False)
            ensemble = compute_ensemble_prob(full_df)
            result["ensemble"] = ensemble
        except Exception:
            pass

        # Store in database
        try:
            prediction_record = Prediction.from_request(
                data, result, warnings, request.remote_addr
            )
            db.session.add(prediction_record)
            db.session.commit()

            result["prediction_id"] = prediction_record.id

            # Update daily statistics
            update_daily_stats(result)

            # Clear cached responses that depend on prediction history
            try:
                cache.delete_memoized("statistics")
            except Exception:
                pass
            try:
                cache.delete_memoized("analytics")
            except Exception:
                pass
            try:
                cache.delete_memoized("home")
            except Exception:
                pass
            # As a fallback, clear the entire cache to ensure clients see
            # updated statistics immediately. This is acceptable for
            # development; in production consider a shared cache backend
            # and more selective invalidation.
            try:
                cache.clear()
                logger.info("Cache cleared after storing prediction")
            except Exception:
                logger.debug("Could not clear cache after prediction")

            logger.info(
                f"Prediction stored: ID={prediction_record.id}, Result={result['prediction']}"
            )
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            result["database_warning"] = "Prediction not stored in database"

        logger.info(
            f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2%}"
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error", "message": str(e)}), 500


@app.route("/models")
def list_models():
    """
    List all available models
    ---
    tags:
      - Models
    responses:
      200:
        description: List of available models
    """
    models_info = {}

    # Try reloading model_comparison from disk so the UI shows latest training results
    comp = model_comparison
    try:
        comp_path = os.path.join("models", "model_comparison.json")
        if os.path.exists(comp_path):
            with open(comp_path, "r", encoding="utf-8") as f:
                comp = json.load(f)
    except Exception:
        comp = model_comparison

    # Define explicit display order so frontend layout is predictable.
    # We want Random Forest to appear in the middle (second card in a 3-col grid),
    # so list models in the order below. If a model is not available, skip it.
    display_order = [
        "gradient_boosting",
        "random_forest",
        "logistic_regression",
        "support_vector_machine",
    ]

    for model_name in display_order:
        if model_name not in available_models:
            continue
        model_key = model_name.replace("_", " ").title()
        model_data = comp.get("models", {}).get(model_key, {})
        models_info[model_name] = {
            "name": model_key,
            "loaded": True,
            "accuracy": model_data.get("accuracy"),
            "precision": model_data.get("precision"),
            "recall": model_data.get("recall"),
            "f1_score": model_data.get("f1_score"),
            "training_time": model_data.get("training_time"),
            "avg_prediction_time": model_data.get("avg_prediction_time"),
        }

    best_model_raw = comp.get("best_model", "") or ""
    best_model_key = best_model_raw.lower().replace(" ", "_")

    # Clear cached view for models so frontend sees latest comp file immediately
    try:
        cache.delete_memoized(list_models)
    except Exception:
        pass

    return jsonify(
        {
            "available_models": list(available_models.keys()),
            "model_count": len(available_models),
            "default_model": "random_forest",
            "best_model": best_model_key,
            "models": models_info,
            # Provide an ordered list representation so clients can rely on display order
            "models_list": [
                {"id": name, **models_info[name]} for name in models_info.keys()
            ],
        }
    )


@app.route("/models/compare")
@cache.cached(timeout=300)
def compare_models():
    """
    Compare all models performance
    ---
    tags:
      - Models
    responses:
      200:
        description: Model comparison data
    """
    # Try to reload model comparison from disk so UI shows freshest results
    try:
        comp_path = os.path.join("models", "model_comparison.json")
        if os.path.exists(comp_path):
            with open(comp_path, "r", encoding="utf-8") as f:
                comp = json.load(f)
            try:
                cache.delete_memoized(compare_models)
            except Exception:
                pass
            return jsonify(comp)
    except Exception:
        pass

    if not model_comparison:
        return jsonify({"error": "Model comparison data not available"}), 404

    return jsonify(model_comparison)


@app.route("/predict/<model_name>", methods=["POST"])
@limiter.limit("100 per hour")
def predict_with_model(model_name):
    """Make prediction with specific model
    ---
    tags:
      - Prediction
    parameters:
      - name: model_name
        in: path
        type: string
        required: true
        description: >-
          Model to use (random_forest, logistic_regression,
          gradient_boosting, support_vector_machine)
      - name: body
        in: body
        required: true
        schema:
          type: object
    responses:
      200:
        description: Prediction result
      400:
        description: Invalid model or data
    """
    try:
        if model_name not in available_models:
            return (
                jsonify(
                    {
                        "error": f"Model not found: {model_name}",
                        "available_models": list(available_models.keys()),
                    }
                ),
                400,
            )

        selected_model = calibrated_models.get(model_name, available_models[model_name])

        data = request.get_json(silent=True)
        # Allow empty body for benchmarking; use defaults in preprocessing
        if data is None:
            data = {}

        # For benchmarking we allow defaults and therefore skip strict validation
        warnings = []

        # Preprocess full engineered dataframe and pick model-specific features
        full_df = preprocess_input(data, select_features=False)

        # Decide whether to pass raw inputs (for pipelines) or engineered features
        raw_fields = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area",
        ]
        raw_df = full_df.copy()
        # Ensure raw-like columns exist and have defaults
        for col in raw_fields:
            if col not in raw_df.columns:
                # Use same defaults as preprocessing
                if col == "CoapplicantIncome":
                    raw_df[col] = 0
                elif col == "LoanAmount":
                    raw_df[col] = 128.0
                elif col == "Loan_Amount_Term":
                    raw_df[col] = 360.0
                elif col == "Credit_History":
                    raw_df[col] = 1.0
                elif col == "Dependents":
                    raw_df[col] = "0"
                elif col == "Gender":
                    raw_df[col] = "Male"
                elif col == "Married":
                    raw_df[col] = "Yes"
                elif col == "Education":
                    raw_df[col] = "Graduate"
                elif col == "Self_Employed":
                    raw_df[col] = "No"
                elif col == "Property_Area":
                    raw_df[col] = "Urban"
                else:
                    raw_df[col] = 0

        # If model appears to be a pipeline (has steps), give it the raw input and let it transform
        model_is_pipeline = hasattr(selected_model, "steps") and isinstance(
            getattr(selected_model, "steps"), list
        )

        try:
            fnames = getattr(selected_model, "feature_names_in_", None)
            if fnames is not None and len(fnames) > 0:
                expected = list(fnames)
                logger.info(
                    "[DIAG] Using model's own feature_names_in_ for prediction: %d features",
                    len(expected),
                )
            else:
                expected = feature_names
                logger.info(
                    "[DIAG] Falling back to global feature_names: %d features",
                    len(expected),
                )
        except Exception:
            expected = feature_names
            logger.info(
                "[DIAG] Exception reading feature_names_in_; falling back to global (%d)",
                len(expected),
            )

        if model_is_pipeline:
            df_to_predict = raw_df[raw_fields].copy()
        else:
            # Build model-specific input (map/compute columns and order exactly as expected)
            df_to_predict = build_model_input(full_df, expected)

        # Diagnostic: log dtypes and sample values before prediction
        try:
            logger.info(
                "[PRED_DIAG] Predicting with model=%s selected_model_type=%s",
                model_name,
                type(selected_model),
            )
            logger.info(
                "[PRED_DIAG] df_to_predict dtypes: %s",
                dict(df_to_predict.dtypes.apply(lambda x: x.name)),
            )
            logger.info(
                "[PRED_DIAG] df_to_predict sample: %s",
                df_to_predict.iloc[0].to_dict(),
            )
        except Exception:
            logger.debug("[PRED_DIAG] Could not serialize df_to_predict diagnostics")

        prediction = selected_model.predict(df_to_predict)[0]
        prediction_proba = selected_model.predict_proba(df_to_predict)[0]

        # Get model info
        model_data = model_comparison.get("models", {}).get(
            model_name.replace("_", " ").title(), {}
        )

        result = {
            "success": True,
            "model_used": model_name,
            "model_name": model_name.replace("_", " ").title(),
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "prediction_code": int(prediction),
            "confidence": float(max(prediction_proba)),
            "probability": {
                "rejected": float(prediction_proba[0]),
                "approved": float(prediction_proba[1]),
            },
            "model_info": {
                "accuracy": model_data.get("accuracy"),
                "precision": model_data.get("precision"),
                "recall": model_data.get("recall"),
                "f1_score": model_data.get("f1_score"),
            },
            "timestamp": datetime.now().isoformat(),
        }

        if warnings:
            result["warnings"] = warnings

        logger.info(f"Prediction with {model_name}: {result['prediction']}")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/models/benchmark", methods=["POST"])
@limiter.limit("10 per hour")
def benchmark_models():
    """
    Benchmark all models with same input
    ---
    tags:
      - Models
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
    responses:
      200:
        description: Benchmark results from all models
    """
    try:
        data = request.get_json(silent=True)
        # Allow empty body for benchmarking; use defaults in preprocessing
        if data is None:
            data = {}

        # For benchmarking we allow defaults and therefore skip strict validation

        # Preprocess once (full engineered dataframe)
        full_df = preprocess_input(data, select_features=False)

        # Test all models
        results = {}
        for model_name, model in available_models.items():
            start_time = time.time()

            # Prepare raw and engineered inputs
            raw_fields = [
                "ApplicantIncome",
                "CoapplicantIncome",
                "LoanAmount",
                "Loan_Amount_Term",
                "Credit_History",
                "Gender",
                "Married",
                "Dependents",
                "Education",
                "Self_Employed",
                "Property_Area",
            ]
            raw_df = full_df.copy()
            for col in raw_fields:
                if col not in raw_df.columns:
                    if col == "CoapplicantIncome":
                        raw_df[col] = 0
                    elif col == "LoanAmount":
                        raw_df[col] = 128.0
                    elif col == "Loan_Amount_Term":
                        raw_df[col] = 360.0
                    elif col == "Credit_History":
                        raw_df[col] = 1.0
                    elif col == "Dependents":
                        raw_df[col] = "0"
                    elif col == "Gender":
                        raw_df[col] = "Male"
                    elif col == "Married":
                        raw_df[col] = "Yes"
                    elif col == "Education":
                        raw_df[col] = "Graduate"
                    elif col == "Self_Employed":
                        raw_df[col] = "No"
                    elif col == "Property_Area":
                        raw_df[col] = "Urban"
                    else:
                        raw_df[col] = 0

            model_is_pipeline = hasattr(model, "steps") and isinstance(
                getattr(model, "steps"), list
            )

            try:
                fnames = getattr(model, "feature_names_in_", None)
                if fnames is not None and len(fnames) > 0:
                    expected = list(fnames)
                    logger.info(
                        "[DIAG] Using %s feature_names_in_: %d features",
                        model_name,
                        len(expected),
                    )
                else:
                    expected = feature_names
                    logger.info(
                        "[DIAG] Using global feature_names for %s: %d features",
                        model_name,
                        len(expected),
                    )
            except Exception:
                expected = feature_names
                logger.info(
                    "[DIAG] Could not read feature_names_in_; using global: %d features",
                    len(expected),
                )

            if model_is_pipeline:
                df_to_predict = raw_df[raw_fields].copy()
            else:
                df_to_predict = build_model_input(full_df, expected)

            # Diagnostic logging before prediction
            try:
                logger.info(
                    "[DIAG] Benchmark model=%s, pipeline=%s",
                    model_name,
                    model_is_pipeline,
                )
                count_expected = (
                    len(expected) if not model_is_pipeline else len(raw_fields)
                )
                logger.info("[DIAG] expected_count=%d", count_expected)
                cols = list(df_to_predict.columns)
                logger.info("[DIAG] columns sent: %s", cols)
            except Exception:
                logger.debug(
                    "[DIAG] Could not log diagnostic info for model %s", model_name
                )

            model_used = calibrated_models.get(model_name, None)
            # Always use the original loaded model for the raw prediction to avoid
            # shape-mismatch when a calibrator (e.g., a 1-D Platt LogisticRegression)
            # was erroneously used as a drop-in model.
            try:
                logger.info(
                    "[BMK_DIAG] Benchmarking model=%s, base_type=%s, calibrator_type=%s",
                    model_name,
                    type(model),
                    type(model_used),
                )
                dtypes = dict(df_to_predict.dtypes.apply(lambda x: x.name))
                logger.info("[BMK_DIAG] df_to_predict dtypes: %s", dtypes)
                sample = df_to_predict.iloc[0].to_dict()
                logger.info("[BMK_DIAG] df_to_predict sample: %s", sample)
            except Exception:
                logger.debug(
                    "[BMK_DIAG] Could not serialize df_to_predict diagnostics for benchmark"
                )

            # Use base model for prediction
            try:
                prediction = model.predict(df_to_predict)[0]
            except Exception as e:
                logger.error(
                    "[BMK_DIAG] Base model.predict failed for %s: %s",
                    model_name,
                    e,
                    exc_info=True,
                )
                prediction = 0

            # Resolve probabilities: prefer using calibrator if available, but handle
            # platt (1-D) calibrators by computing base raw proba first.
            try:
                if model_used is not None:
                    # If calibrator exposes predict_proba that accepts full X
                    # (CalibratedClassifierCV or wrapper)
                    try:
                        prediction_proba = model_used.predict_proba(df_to_predict)[0]
                    except Exception:
                        # Fallback: compute raw probabilities from base model and feed to calibrator
                        raw_proba = model.predict_proba(df_to_predict)[:, 1]
                        prediction_proba = model_used.predict_proba(
                            raw_proba.reshape(-1, 1)
                        )[0]
                else:
                    prediction_proba = model.predict_proba(df_to_predict)[0]
            except Exception as e:
                logger.error(
                    "[BMK_DIAG] Probability computation failed for %s: %s",
                    model_name,
                    e,
                    exc_info=True,
                )
                prediction_proba = [0.5, 0.5]
            prediction_time = time.time() - start_time

            results[model_name] = {
                "prediction": "Approved" if prediction == 1 else "Rejected",
                "confidence": float(max(prediction_proba)),
                "probability": {
                    "rejected": float(prediction_proba[0]),
                    "approved": float(prediction_proba[1]),
                },
                "prediction_time": f"{prediction_time*1000:.2f}ms",
            }

        # Find consensus
        predictions = [r["prediction"] for r in results.values()]
        consensus = max(set(predictions), key=predictions.count)
        agreement = predictions.count(consensus) / len(predictions) * 100

        return (
            jsonify(
                {
                    "input_data": data,
                    "results": results,
                    "consensus": {
                        "prediction": consensus,
                        "agreement": f"{agreement:.1f}%",
                        "models_agree": predictions.count(consensus),
                        "total_models": len(predictions),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Benchmark error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/validate-loan", methods=["POST"])
def validate_loan():
    """Validate loan application data before prediction"""
    logger.info(f"Validation request received from {request.remote_addr}")

    try:
        try:
            data = request.get_json(force=True)
        except Exception as e:
            return (
                jsonify(
                    {
                        "error": "Invalid JSON in request body",
                        "message": "Please ensure your request body contains valid JSON",
                        "details": str(e),
                    }
                ),
                400,
            )

        if not data:
            return (
                jsonify(
                    {
                        "error": "No data provided",
                        "message": "Please send JSON data in request body",
                    }
                ),
                400,
            )

        is_valid, errors, warnings = validator.validate_loan_application(data)

        response = {
            "valid": is_valid,
            "timestamp": datetime.now().isoformat(),
            "input_data": data,
        }

        if errors:
            response["errors"] = errors
            response["message"] = "Validation failed - please correct the errors"

        if warnings:
            response["warnings"] = warnings

        if is_valid and not warnings:
            response["message"] = "Data is valid and ready for prediction"
        elif is_valid and warnings:
            response["message"] = "Data is valid but has warnings"

        if is_valid:
            logger.info(f"Validation successful with {len(warnings)} warnings")
        else:
            logger.warning(f"Validation failed: {errors}")

        status_code = 200 if is_valid else 400
        return jsonify(response), status_code

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return (
            jsonify({"error": "Server error during validation", "message": str(e)}),
            500,
        )


@app.route("/history")
@swag_from(os.path.join(os.path.dirname(__file__), "docs", "swagger", "history.yml"))
@cache.cached(timeout=30, query_string=True)
def history():
    """Get recent prediction history"""
    try:
        limit = request.args.get("limit", 10, type=int)
        limit = min(limit, 100)

        predictions = get_recent_predictions(limit)

        return (
            jsonify(
                {
                    "count": len(predictions),
                    "predictions": [p.to_dict() for p in predictions],
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/history/<int:prediction_id>")
@swag_from(os.path.join(os.path.dirname(__file__), "docs", "swagger", "history_id.yml"))
@cache.cached(timeout=300)
def get_prediction(prediction_id):
    """Get specific prediction by ID"""
    try:
        prediction = Prediction.query.get(prediction_id)

        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404

        return jsonify(prediction.to_dict()), 200
    except Exception as e:
        logger.error(f"Get prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/statistics")
@swag_from(os.path.join(os.path.dirname(__file__), "docs", "swagger", "statistics.yml"))
@cache.cached(timeout=60)
def statistics():
    """Get overall statistics"""
    try:
        stats = get_statistics()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Statistics error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/analytics")
@swag_from(os.path.join(os.path.dirname(__file__), "docs", "swagger", "analytics.yml"))
@cache.cached(timeout=120)
def analytics():
    """Get detailed analytics"""
    try:
        stats = get_statistics()

        yesterday = datetime.now() - timedelta(days=1)
        recent = Prediction.query.filter(Prediction.timestamp >= yesterday).all()

        recent_approved = sum(1 for p in recent if p.prediction == "Approved")
        recent_rejected = len(recent) - recent_approved

        approved_predictions = Prediction.query.filter_by(prediction="Approved").all()
        rejected_predictions = Prediction.query.filter_by(prediction="Rejected").all()

        avg_confidence_approved = (
            np.mean([p.confidence for p in approved_predictions])
            if approved_predictions
            else 0
        )
        avg_confidence_rejected = (
            np.mean([p.confidence for p in rejected_predictions])
            if rejected_predictions
            else 0
        )

        return (
            jsonify(
                {
                    "overall": stats,
                    "last_24_hours": {
                        "total": len(recent),
                        "approved": recent_approved,
                        "rejected": recent_rejected,
                        "approval_rate": (
                            f"{(recent_approved / len(recent) * 100):.2f}%"
                            if recent
                            else "0%"
                        ),
                    },
                    "confidence_analysis": {
                        "avg_confidence_approved": f"{avg_confidence_approved:.2%}",
                        "avg_confidence_rejected": f"{avg_confidence_rejected:.2%}",
                    },
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/model-info")
@swag_from(os.path.join(os.path.dirname(__file__), "docs", "swagger", "model_info.yml"))
@cache.cached(timeout=3600)
def model_info_endpoint():
    """Return model information"""
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    return jsonify(
        {
            "accuracy": model_info.get("accuracy"),
            "precision": model_info.get("precision"),
            "recall": model_info.get("recall"),
            "f1_score": model_info.get("f1_score"),
            "features": feature_names,
            "feature_count": len(feature_names),
        }
    )


@app.route("/validation-rules")
@swag_from(
    os.path.join(os.path.dirname(__file__), "docs", "swagger", "validation_rules.yml")
)
@cache.cached(timeout=3600)
def validation_rules():
    """Return validation rules"""
    return jsonify(
        {
            "required_fields": ["ApplicantIncome"],
            "optional_fields": [
                "CoapplicantIncome",
                "LoanAmount",
                "Loan_Amount_Term",
                "Credit_History",
                "Gender",
                "Married",
                "Dependents",
                "Education",
                "Self_Employed",
                "Property_Area",
            ],
            "valid_values": {
                "Gender": validator.VALID_GENDER,
                "Married": validator.VALID_MARRIED,
                "Dependents": validator.VALID_DEPENDENTS,
                "Education": validator.VALID_EDUCATION,
                "Self_Employed": validator.VALID_SELF_EMPLOYED,
                "Property_Area": validator.VALID_PROPERTY_AREA,
            },
            "ranges": {
                "ApplicantIncome": f"{validator.MIN_INCOME} - {validator.MAX_INCOME}",
                "CoapplicantIncome": f"{validator.MIN_INCOME} - {validator.MAX_INCOME}",
                "LoanAmount": f"{validator.MIN_LOAN_AMOUNT} - {validator.MAX_LOAN_AMOUNT}",
                "Credit_History": "0 or 1",
                "Loan_Amount_Term": validator.VALID_LOAN_TERMS,
            },
            "example_request": {
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 1500,
                "LoanAmount": 150,
                "Loan_Amount_Term": 360,
                "Credit_History": 1,
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "Property_Area": "Urban",
            },
        }
    )


@app.route("/performance")
def performance_metrics():
    """
    Get performance metrics
    ---
    tags:
      - Information
    responses:
      200:
        description: Performance metrics
    """
    if not request_times:
        return jsonify(
            {
                "message": "No requests tracked yet",
                "avg_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0,
            }
        )

    return jsonify(
        {
            "requests_tracked": len(request_times),
            "avg_response_time": f"{np.mean(request_times):.3f}s",
            "min_response_time": f"{min(request_times):.3f}s",
            "max_response_time": f"{max(request_times):.3f}s",
            "median_response_time": f"{np.median(request_times):.3f}s",
            "p95_response_time": f"{np.percentile(request_times, 95):.3f}s",
            "p99_response_time": f"{np.percentile(request_times, 99):.3f}s",
            "cache_info": {
                "type": app.config["CACHE_TYPE"],
                "timeout": app.config["CACHE_DEFAULT_TIMEOUT"],
            },
            "rate_limits": {
                "default": "200 per hour, 1000 per day",
                "predictions": "100 per hour",
            },
            "compression": "enabled",
        }
    )


@app.route("/search", methods=["GET"])
def search():
    """Example endpoint for query parameters"""
    query = request.args.get("q")
    limit = request.args.get("limit", default=10, type=int)
    page = request.args.get("page", 1, type=int)

    return jsonify(
        {
            "query": query,
            "limit": limit,
            "page": page,
            "message": f"Searching for '{query}' with limit {limit} on page {page}",
        }
    )


@app.route("/headers", methods=["GET", "POST"])
def show_headers():
    """Show request headers (for debugging)"""
    user_agent = request.headers.get("User-Agent")
    content_type = request.headers.get("Content-Type")
    all_headers = dict(request.headers)

    return jsonify(
        {
            "user_agent": user_agent,
            "content_type": content_type,
            "all_headers": all_headers,
        }
    )


@app.route("/request-info", methods=["GET", "POST"])
def request_info():
    """Show request metadata (for debugging)"""
    return jsonify(
        {
            "method": request.method,
            "url": request.url,
            "path": request.path,
            "remote_addr": request.remote_addr,
            "is_json": request.is_json,
            "content_type": request.content_type,
            "content_length": request.content_length,
        }
    )


@app.route("/cache/clear", methods=["POST"])
@limiter.limit("10 per hour")
def clear_cache():
    """
    Clear all caches (admin endpoint)
    ---
    tags:
      - Information
    responses:
      200:
        description: Cache cleared
    """
    cache.clear()
    return jsonify(
        {
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/admin/calibrate", methods=["POST"])
@limiter.limit("2 per hour")
def admin_calibrate():
    """Trigger calibration of available models (admin only).
    This runs `calibrate_models()` in the server process so calibrated
    wrappers can be used immediately by prediction endpoints.
    """
    try:
        calibrate_models()
        return (
            jsonify(
                {
                    "status": "ok",
                    "message": "Calibration attempted",
                    "ensemble_weights": model_comparison.get("ensemble_weights", {}),
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Calibration endpoint error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return (
        jsonify(
            {
                "error": "Endpoint not found",
                "message": "The requested endpoint does not exist",
                "available_endpoints": [
                    "/",
                    "/app",
                    "/health",
                    "/predict",
                    "/predict/<model>",
                    "/validate-loan",
                    "/models",
                    "/models/compare",
                    "/models/benchmark",
                    "/history",
                    "/statistics",
                    "/analytics",
                    "/model-info",
                    "/validation-rules",
                    "/performance",
                    "/search",
                    "/headers",
                    "/request-info",
                    "/cache/clear",
                    "/docs",
                ],
            }
        ),
        404,
    )


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return (
        jsonify(
            {
                "error": "Method not allowed",
                "message": "This endpoint does not support the requested HTTP method",
            }
        ),
        405,
    )


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return (
        jsonify(
            {
                "error": "Internal server error",
                "message": "An unexpected error occurred",
            }
        ),
        500,
    )


def load_model(default_model_name="random_forest"):
    """
    Load available models (using load_all_models), select a default model to use
    for single-model endpoints, and initialize feature_names and model_info.
    Returns True if a model was selected, False otherwise.
    """
    global model

    # Ensure models are loaded into available_models
    try:
        load_all_models()
    except Exception as e:
        logger.warning(f"[LOAD] load_all_models() failed: {e}")

    # Select default model if available, otherwise pick any loaded model
    if default_model_name in available_models:
        model_key = default_model_name
        model_obj = available_models[default_model_name]
    elif available_models:
        model_key, model_obj = next(iter(available_models.items()))
        logger.info(
            f"[LOAD] Default model '{default_model_name}' not found; using '{model_key}' instead"
        )
    else:
        logger.warning("[LOAD] No models available to load as default")
        return False

    model = model_obj

    # Try to infer feature names from the model or from a saved file
    try:
        fnames = getattr(model, "feature_names_in_", None)
        if fnames is not None and len(fnames) > 0:
            feature_names[:] = list(fnames)
        else:
            # fallback: try to load a serialized feature list
            features_path = os.path.join("models", "feature_names.json")
            if os.path.exists(features_path):
                try:
                    with open(features_path, "r") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, list):
                            feature_names[:] = loaded
                except Exception:
                    logger.debug("[LOAD] Could not read models/feature_names.json")
            # if still empty, leave feature_names as-is (may be populated elsewhere)
    except Exception as e:
        logger.warning(f"[LOAD] Error while setting feature names: {e}")

    # Populate model_info from model_comparison if possible
    try:
        comp_key = model_key.replace("_", " ").title()
        info = model_comparison.get("models", {}).get(comp_key, {})
        model_info.update(
            {
                "name": model_key,
                "accuracy": info.get("accuracy"),
                "precision": info.get("precision"),
                "recall": info.get("recall"),
                "f1_score": info.get("f1_score"),
            }
        )
    except Exception:
        logger.debug("[LOAD] Could not populate model_info from model_comparison")

    logger.info(
        f"[LOAD] Selected default model: {model_key}, features: {len(feature_names)}"
    )
    return True


# Load model on startup
with app.app_context():
    if not load_model():
        logger.warning("⚠️  [WARNING] Model failed to load")

if __name__ == "__main__":
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get("PORT", 5000))

    # Run in production mode
    # If running locally (no PORT env var or default 5000), open browser to /app
    try:
        if os.environ.get("PORT") is None or port == 5000:

            def _open():
                webbrowser.open(f"http://127.0.0.1:{port}/app")

            threading.Timer(1.0, _open).start()
    except Exception:
        pass

    # Optional SSL: if `SSL_CERT` and `SSL_KEY` environment variables are
    # set to paths of a certificate and private key, Flask will run with
    # an SSL context so the API is available via HTTPS locally.
    ssl_cert = os.environ.get("SSL_CERT")
    ssl_key = os.environ.get("SSL_KEY")
    ssl_context = None
    if ssl_cert and ssl_key and os.path.exists(ssl_cert) and os.path.exists(ssl_key):
        ssl_context = (ssl_cert, ssl_key)
        logger.info(f"Starting server with SSL context: cert={ssl_cert}, key={ssl_key}")
    else:
        if ssl_cert or ssl_key:
            logger.warning(
                "SSL_CERT or SSL_KEY specified but files not found; starting without SSL"
            )

    app.run(host="0.0.0.0", port=port, debug=False, ssl_context=ssl_context)
