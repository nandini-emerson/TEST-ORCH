
# python

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("attendance_config")

# --- API Key Management ---
def get_api_key(key_name, required=True):
    value = os.getenv(key_name)
    if required and not value:
        logger.error(f"Missing required API key: {key_name}")
        raise RuntimeError(f"Missing required API key: {key_name}")
    return value

# --- LLM Configuration ---
LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "azure"),
    "model": os.getenv("LLM_MODEL", "gpt-4.1"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
    "system_prompt": os.getenv("LLM_SYSTEM_PROMPT", "You are a professional Student Attendance Management Agent. Your role is to assist authorized users in recording, updating, and reporting student attendance. Always validate user permissions, ensure data accuracy, and maintain compliance with privacy and audit requirements."),
    "user_prompt_template": os.getenv("LLM_USER_PROMPT_TEMPLATE", "Please provide the student ID, date, and attendance status (Present, Absent, Late) to record attendance."),
    "few_shot_examples": [
        "Record attendance for student 12345 on 2024-06-10 as Present.",
        "Update attendance for student 67890 on 2024-06-09 to Late."
    ],
    "model_details": {
        "provider": os.getenv("LLM_PROVIDER", "azure"),
        "model": os.getenv("LLM_MODEL", "gpt-4.1"),
        "fallback_model": os.getenv("LLM_FALLBACK_MODEL", "gpt-4.1")
    }
}

# --- Domain-specific Settings ---
DOMAIN = os.getenv("AGENT_DOMAIN", "general")
AGENT_NAME = os.getenv("AGENT_NAME", "Student Attendance Management Agent")

# --- API Endpoints and Authentication ---
ATTENDANCE_API_URL = os.getenv("ATTENDANCE_API_URL", "https://attendance.example.com/api")
ATTENDANCE_API_KEY = get_api_key("ATTENDANCE_API_KEY")
USER_DIRECTORY_API_URL = os.getenv("USER_DIRECTORY_API_URL", "https://userdir.example.com/api")
USER_DIRECTORY_OAUTH_TOKEN = get_api_key("USER_DIRECTORY_OAUTH_TOKEN", required=False)
AUDIT_LOG_API_URL = os.getenv("AUDIT_LOG_API_URL", "https://auditlog.example.com/api")
AUDIT_LOG_OAUTH_TOKEN = get_api_key("AUDIT_LOG_OAUTH_TOKEN", required=False)

API_CONFIG = {
    "attendance_api": {
        "url": ATTENDANCE_API_URL,
        "key": ATTENDANCE_API_KEY,
        "auth_type": "API Key",
        "rate_limit": int(os.getenv("ATTENDANCE_API_RATE_LIMIT", "100"))
    },
    "user_directory_api": {
        "url": USER_DIRECTORY_API_URL,
        "oauth_token": USER_DIRECTORY_OAUTH_TOKEN,
        "auth_type": "OAuth 2.0",
        "rate_limit": int(os.getenv("USER_DIRECTORY_API_RATE_LIMIT", "50"))
    },
    "audit_log_api": {
        "url": AUDIT_LOG_API_URL,
        "oauth_token": AUDIT_LOG_OAUTH_TOKEN,
        "auth_type": "OAuth 2.0",
        "rate_limit": os.getenv("AUDIT_LOG_API_RATE_LIMIT", "unlimited")
    }
}

# --- Validation and Error Handling ---
REQUIRED_CONFIG_KEYS = [
    "ATTENDANCE_API_URL",
    "ATTENDANCE_API_KEY"
]

def validate_config():
    missing = []
    for key in REQUIRED_CONFIG_KEYS:
        if not os.getenv(key):
            missing.append(key)
    if missing:
        logger.error(f"Missing required config keys: {missing}")
        raise RuntimeError(f"Missing required config keys: {missing}")

try:
    validate_config()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    raise

# --- Default Values and Fallbacks ---
DEFAULTS = {
    "attendance_status": "Absent",
    "llm_model": "gpt-4.1",
    "domain": "general"
}

# --- Error Codes ---
ERROR_CODES = {
    "ERR_UNAUTHORIZED": "User is not authorized to perform this action.",
    "ERR_INVALID_INPUT": "Input validation failed.",
    "ERR_LOGGING_FAILED": "Audit logging failed.",
    "ERR_API": "Attendance API error.",
    "ERR_INTERNAL": "Internal agent error."
}

# --- Compliance and Security ---
COMPLIANCE = {
    "privacy": "FERPA",
    "data_retention_policy": "institutional",
    "audit_logging": True,
    "pii_masking": True
}

SECURITY = {
    "encryption": "AES-256",
    "authentication": "OAuth 2.0",
    "role_based_access": True,
    "session_management": True
}

# --- Exported Config ---
CONFIG = {
    "agent_name": AGENT_NAME,
    "domain": DOMAIN,
    "llm_config": LLM_CONFIG,
    "api_config": API_CONFIG,
    "defaults": DEFAULTS,
    "error_codes": ERROR_CODES,
    "compliance": COMPLIANCE,
    "security": SECURITY
}
