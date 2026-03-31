try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 4,
 'runtime_enabled': True,
 'sanitize_pii': True}


import os
import logging
import time as _time
from typing import Optional, Dict, Any, List, Union
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field, model_validator
from dotenv import load_dotenv
import requests
from cachetools import TTLCache
from email_validator import validate_email, EmailNotValidError

# Observability wrappers (trace_step, trace_step_sync, etc.) are injected automatically by the runtime.

# Load .env if present
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("attendance_agent")

# --- Configuration Management ---

class Config:
    """Configuration loader for environment variables."""
    @staticmethod
    def get(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(key, default)

    @staticmethod
    def validate(required_keys: List[str]) -> Dict[str, str]:
        """Validate required config keys. Returns missing keys."""
        missing = {}
        for k in required_keys:
            v = os.getenv(k)
            if not v:
                missing[k] = None
        return missing

# --- Models ---

class AttendanceInput(BaseModel):
    input_text: str = Field(..., max_length=50000)
    user_id: str = Field(..., min_length=1, max_length=128)
    communication_channel: Optional[str] = Field(default="chat", max_length=32)

    @field_validator("input_text")
    @classmethod
    def validate_input_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Input text cannot be empty.")
        if len(v) > 50000:
            raise ValueError("Input text exceeds 50,000 characters.")
        return v.strip()

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError("User ID cannot be empty.")
        return v.strip()

class AttendanceRecordInput(BaseModel):
    student_id: str = Field(..., min_length=1, max_length=32)
    date: str = Field(..., min_length=8, max_length=10)  # YYYY-MM-DD
    status: str = Field(..., min_length=1, max_length=16)
    user_id: str = Field(..., min_length=1, max_length=128)

    @field_validator("student_id")
    @classmethod
    def validate_student_id(cls, v):
        return v.strip().upper()

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        allowed = {"PRESENT", "ABSENT", "LATE"}
        val = v.strip().upper()
        if val not in allowed:
            raise ValueError(f"Status must be one of {allowed}")
        return val.capitalize()

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        # Simple YYYY-MM-DD check
        import re
        if not re.match(r"\d{4}-\d{2}-\d{2}", v.strip()):
            raise ValueError("Date must be in YYYY-MM-DD format.")
        return v.strip()

class AttendanceUpdateInput(BaseModel):
    student_id: str = Field(..., min_length=1, max_length=32)
    date: str = Field(..., min_length=8, max_length=10)
    new_status: str = Field(..., min_length=1, max_length=16)
    user_id: str = Field(..., min_length=1, max_length=128)

    @field_validator("new_status")
    @classmethod
    def validate_new_status(cls, v):
        allowed = {"PRESENT", "ABSENT", "LATE"}
        val = v.strip().upper()
        if val not in allowed:
            raise ValueError(f"Status must be one of {allowed}")
        return val.capitalize()

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        import re
        if not re.match(r"\d{4}-\d{2}-\d{2}", v.strip()):
            raise ValueError("Date must be in YYYY-MM-DD format.")
        return v.strip()

class ReportRequestInput(BaseModel):
    report_type: str = Field(..., min_length=3, max_length=32)
    filters: Dict[str, Any]
    user_id: str = Field(..., min_length=1, max_length=128)

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, v):
        allowed = {"student", "class", "date_range"}
        if v.lower() not in allowed:
            raise ValueError(f"Report type must be one of {allowed}")
        return v.lower()

class AttendanceQueryInput(BaseModel):
    student_id: Optional[str] = Field(default=None, max_length=32)
    class_id: Optional[str] = Field(default=None, max_length=32)
    date: Optional[str] = Field(default=None, max_length=10)
    user_id: str = Field(..., min_length=1, max_length=128)

    @model_validator(mode="after")
    def at_least_one_id(self):
        if not self.student_id and not self.class_id:
            raise ValueError("At least one of student_id or class_id must be provided.")
        return self

# --- Utility Functions ---

@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_pii(data: str) -> str:
    """Mask PII in logs (simple version)."""
    import re
    # Mask emails
    data = re.sub(r"([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", "***@***", data)
    # Mask student IDs (assume numeric or alphanumeric)
    data = re.sub(r"\b\d{5,}\b", "*****", data)
    return data

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_text(text: str) -> str:
    """Sanitize input text for LLM and logs."""
    return text.replace("\n", " ").replace("\r", " ").strip()

# --- Integration Layer: Azure OpenAI LLM Client ---

class AzureOpenAIClient:
    """Handles lazy initialization and calls to Azure OpenAI."""
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            api_key = Config.get("AZURE_OPENAI_API_KEY")
            endpoint = Config.get("AZURE_OPENAI_ENDPOINT")
            if not api_key or not endpoint:
                raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set.")
            try:
                import openai
                cls._client = openai.AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=endpoint,
                    api_version="2024-02-15-preview"
                )
            except ImportError as e:
                logger.error(f"openai package not installed: {e}")
                raise
        return cls._client

    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def chat_completion(cls, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int, deployment_id: Optional[str] = None) -> str:
        client = cls.get_client()
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if deployment_id:
            params["deployment_id"] = deployment_id
        _t0 = _time.time()
        response = await client.chat.completions.create(**params)
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else ""
            )
        except Exception:
            pass
        return content

# --- Integration Layer: Attendance API Client ---

class AttendanceAPIClient:
    """Handles CRUD operations for attendance records."""
    def __init__(self):
        self.api_url = Config.get("ATTENDANCE_API_URL")
        self.api_key = Config.get("ATTENDANCE_API_KEY")
        if not self.api_url or not self.api_key:
            logger.warning("Attendance API URL or KEY not configured. API calls will fail.")

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _request(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None, retries: int = 3) -> Dict[str, Any]:
        url = f"{self.api_url.rstrip('/')}/{endpoint.lstrip('/')}"
        for attempt in range(retries):
            _t0 = _time.time()
            try:
                resp = requests.request(
                    method=method,
                    url=url,
                    headers=self._headers(),
                    json=data,
                    params=params,
                    timeout=5
                )
                try:
                    trace_tool_call(
                        tool_name=f"AttendanceAPI.{method}",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(resp.status_code),
                        status="success" if resp.ok else "error"
                    )
                except Exception:
                    pass
                if resp.ok:
                    return resp.json()
                else:
                    logger.warning(f"Attendance API error: {resp.status_code} {resp.text}")
            except Exception as e:
                logger.error(f"Attendance API request failed: {e}")
                if attempt < retries - 1:
                    _time.sleep(2 ** attempt)
        raise HTTPException(status_code=502, detail="Attendance API unavailable.")

    def record_attendance(self, student_id: str, date: str, status: str, user_id: str) -> Dict[str, Any]:
        data = {
            "student_id": student_id,
            "date": date,
            "status": status,
            "recorded_by": user_id
        }
        return self._request("POST", "/attendance/record", data=data)

    def update_attendance(self, student_id: str, date: str, new_status: str, user_id: str) -> Dict[str, Any]:
        data = {
            "student_id": student_id,
            "date": date,
            "status": new_status,
            "updated_by": user_id
        }
        return self._request("PUT", "/attendance/update", data=data)

    def get_attendance(self, student_id: Optional[str], class_id: Optional[str], date: Optional[str], user_id: str) -> Dict[str, Any]:
        params = {
            "student_id": student_id,
            "class_id": class_id,
            "date": date,
            "requested_by": user_id
        }
        return self._request("GET", "/attendance/query", params=params)

    def generate_report(self, report_type: str, filters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        data = {
            "report_type": report_type,
            "filters": filters,
            "requested_by": user_id
        }
        return self._request("POST", "/attendance/report", data=data)

# --- Security & Compliance Layer: Authentication Manager ---

class AuthenticationManager:
    """Manages OAuth 2.0 authentication, role-based access control, session management."""
    # For demo, use in-memory cache for user roles
    _user_roles = TTLCache(maxsize=1000, ttl=600)  # 10 min cache

    def __init__(self):
        self.user_directory_url = Config.get("USER_DIRECTORY_API_URL")
        self.oauth_token = Config.get("USER_DIRECTORY_OAUTH_TOKEN")

    def get_user_role(self, user_id: str) -> Optional[str]:
        if user_id in self._user_roles:
            return self._user_roles[user_id]
        # Simulate API call to user directory
        if not self.user_directory_url or not self.oauth_token:
            logger.warning("User Directory API not configured. Defaulting to 'teacher'.")
            role = "teacher"
        else:
            try:
                _obs_t0 = _time.time()
                resp = requests.get(
                    f"{self.user_directory_url.rstrip('/')}/users/{user_id}/role",
                    headers={"Authorization": f"Bearer {self.oauth_token}"},
                    timeout=3
                )
                try:
                    trace_tool_call(
                        tool_name='requests.get',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(resp)[:200] if resp is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                if resp.ok:
                    role = resp.json().get("role", "teacher")
                else:
                    logger.warning(f"User Directory API error: {resp.status_code}")
                    role = "teacher"
            except Exception as e:
                logger.error(f"User Directory API request failed: {e}")
                role = "teacher"
        self._user_roles[user_id] = role
        return role

    def validate_user(self, user_id: str, required_role: str) -> bool:
        role = self.get_user_role(user_id)
        if role is None:
            return False
        allowed_roles = {
            "teacher": ["teacher", "admin"],
            "admin": ["admin"],
            "student": ["student"]
        }
        return role in allowed_roles.get(required_role, [required_role])

# --- Security & Compliance Layer: Audit Logger ---

class AuditLogger:
    """Logs all attendance modifications, tracks user actions, ensures audit compliance."""
    def __init__(self):
        self.audit_log_url = Config.get("AUDIT_LOG_API_URL")
        self.oauth_token = Config.get("AUDIT_LOG_OAUTH_TOKEN")

    def log_audit_entry(self, record_id: str, user_id: str, action: str, timestamp: str) -> bool:
        data = {
            "record_id": record_id,
            "user_id": user_id,
            "action": action,
            "timestamp": timestamp
        }
        if not self.audit_log_url or not self.oauth_token:
            logger.warning("Audit Log API not configured. Skipping audit log.")
            return False
        _t0 = _time.time()
        try:
            resp = requests.post(
                f"{self.audit_log_url.rstrip('/')}/audit/log",
                headers={"Authorization": f"Bearer {self.oauth_token}", "Content-Type": "application/json"},
                json=data,
                timeout=3
            )
            try:
                trace_tool_call(
                    tool_name="AuditLogger.log_audit_entry",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=str(resp.status_code),
                    status="success" if resp.ok else "error"
                )
            except Exception:
                pass
            if resp.ok:
                return True
            else:
                logger.warning(f"Audit Log API error: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Audit Log API request failed: {e}")
            return False

# --- Domain Logic Layer ---

class DomainLogicEngine:
    """Implements attendance business rules, validation, decision tables, mappings."""
    def __init__(self, attendance_api: AttendanceAPIClient, audit_logger: AuditLogger, auth_manager: AuthenticationManager):
        self.attendance_api = attendance_api
        self.audit_logger = audit_logger
        self.auth_manager = auth_manager

    async def record_attendance(self, student_id: str, date: str, status: str, user_id: str) -> Dict[str, Any]:
        async with trace_step(
            "validate_user", step_type="process",
            decision_summary="Check user authorization for attendance recording",
            output_fn=lambda r: f"authorized={r}"
        ) as step:
            authorized = self.auth_manager.validate_user(user_id, "teacher")
            step.capture({"authorized": authorized})
            if not authorized:
                return {
                    "success": False,
                    "error": "ERR_UNAUTHORIZED",
                    "message": "User not authorized to record attendance.",
                    "fix_tip": "Contact your administrator for access."
                }
        async with trace_step(
            "record_attendance_api", step_type="tool_call",
            decision_summary="Call Attendance API to record attendance",
            output_fn=lambda r: f"result={r.get('status','?')}"
        ) as step:
            try:
                result = self.attendance_api.record_attendance(student_id, date, status, user_id)
                step.capture(result)
            except Exception as e:
                logger.error(f"Attendance API error: {e}")
                return {
                    "success": False,
                    "error": "ERR_API",
                    "message": "Failed to record attendance.",
                    "fix_tip": "Try again later or contact support."
                }
        return {
            "success": True,
            "message": f"Attendance recorded for student {student_id} on {date} as {status}."
        }

    async def update_attendance(self, student_id: str, date: str, new_status: str, user_id: str) -> Dict[str, Any]:
        async with trace_step(
            "validate_user", step_type="process",
            decision_summary="Check user authorization for attendance update",
            output_fn=lambda r: f"authorized={r}"
        ) as step:
            authorized = self.auth_manager.validate_user(user_id, "teacher")
            step.capture({"authorized": authorized})
            if not authorized:
                return {
                    "success": False,
                    "error": "ERR_UNAUTHORIZED",
                    "message": "User not authorized to update attendance.",
                    "fix_tip": "Contact your administrator for access."
                }
        async with trace_step(
            "update_attendance_api", step_type="tool_call",
            decision_summary="Call Attendance API to update attendance",
            output_fn=lambda r: f"result={r.get('status','?')}"
        ) as step:
            try:
                result = self.attendance_api.update_attendance(student_id, date, new_status, user_id)
                step.capture(result)
            except Exception as e:
                logger.error(f"Attendance API error: {e}")
                return {
                    "success": False,
                    "error": "ERR_API",
                    "message": "Failed to update attendance.",
                    "fix_tip": "Try again later or contact support."
                }
        # Audit log
        import datetime
        timestamp = datetime.datetime.utcnow().isoformat()
        async with trace_step(
            "log_audit_entry", step_type="tool_call",
            decision_summary="Log attendance modification for compliance",
            output_fn=lambda r: f"logged={r}"
        ) as step:
            logged = self.audit_logger.log_audit_entry(
                record_id=f"{student_id}_{date}",
                user_id=user_id,
                action=f"update:{new_status}",
                timestamp=timestamp
            )
            step.capture({"logged": logged})
            if not logged:
                logger.warning(f"Audit log failed for {student_id} {date}")
        return {
            "success": True,
            "message": f"Attendance updated for student {student_id} on {date} to {new_status}."
        }

    async def generate_report(self, report_type: str, filters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        async with trace_step(
            "validate_user", step_type="process",
            decision_summary="Check user authorization for report generation",
            output_fn=lambda r: f"authorized={r}"
        ) as step:
            authorized = self.auth_manager.validate_user(user_id, "teacher")
            step.capture({"authorized": authorized})
            if not authorized:
                return {
                    "success": False,
                    "error": "ERR_UNAUTHORIZED",
                    "message": "User not authorized to generate reports.",
                    "fix_tip": "Contact your administrator for access."
                }
        async with trace_step(
            "generate_report_api", step_type="tool_call",
            decision_summary="Call Attendance API to generate report",
            output_fn=lambda r: f"result={str(r)[:100]}"
        ) as step:
            try:
                result = self.attendance_api.generate_report(report_type, filters, user_id)
                step.capture(result)
            except Exception as e:
                logger.error(f"Attendance API error: {e}")
                return {
                    "success": False,
                    "error": "ERR_API",
                    "message": "Failed to generate report.",
                    "fix_tip": "Try again later or contact support."
                }
        return {
            "success": True,
            "report": result
        }

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def query_attendance_status(self, student_id: Optional[str], class_id: Optional[str], date: Optional[str], user_id: str) -> Dict[str, Any]:
        async with trace_step(
            "validate_user", step_type="process",
            decision_summary="Check user authorization for attendance query",
            output_fn=lambda r: f"authorized={r}"
        ) as step:
            authorized = self.auth_manager.validate_user(user_id, "teacher")
            step.capture({"authorized": authorized})
            if not authorized:
                return {
                    "success": False,
                    "error": "ERR_UNAUTHORIZED",
                    "message": "User not authorized to query attendance.",
                    "fix_tip": "Contact your administrator for access."
                }
        async with trace_step(
            "query_attendance_api", step_type="tool_call",
            decision_summary="Call Attendance API to query attendance",
            output_fn=lambda r: f"result={str(r)[:100]}"
        ) as step:
            try:
                result = self.attendance_api.get_attendance(student_id, class_id, date, user_id)
                step.capture(result)
            except Exception as e:
                logger.error(f"Attendance API error: {e}")
                return {
                    "success": False,
                    "error": "ERR_API",
                    "message": "Failed to query attendance.",
                    "fix_tip": "Try again later or contact support."
                }
        return {
            "success": True,
            "attendance": result
        }

# --- Application Layer: Application Controller ---

class ApplicationController:
    """Coordinates request processing, invokes domain logic, manages workflow."""
    def __init__(self, domain_engine: DomainLogicEngine, auth_manager: AuthenticationManager):
        self.domain_engine = domain_engine
        self.auth_manager = auth_manager

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_input(self, input_text: str, user_id: str, communication_channel: str) -> Dict[str, Any]:
        async with trace_step(
            "parse_input", step_type="parse",
            decision_summary="Parse user input for intent and entities",
            output_fn=lambda r: f"intent={r.get('intent','?')}"
        ) as step:
            # Use LLM to extract intent and entities
            system_prompt = (
                "You are a professional Student Attendance Management Agent. "
                "Extract the intent (record_attendance, update_attendance, generate_report, query_attendance_status) "
                "and relevant entities (student_id, date, status, class_id, report_type, filters) from the following user message. "
                "Return a JSON object with keys: intent, entities."
            )
            user_prompt = f"User message: {sanitize_text(input_text)}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            try:
                llm_response = await AzureOpenAIClient.chat_completion(
                    messages=messages,
                    model=Config.get("AZURE_OPENAI_MODEL", "gpt-4.1"),
                    temperature=0.2,
                    max_tokens=700
                )
                import json
                parsed = json.loads(llm_response)
                step.capture(parsed)
            except Exception as e:
                logger.error(f"LLM parsing failed: {e}")
                return {
                    "success": False,
                    "error": "ERR_LLM_PARSE",
                    "message": "Could not understand your request. Please rephrase.",
                    "fix_tip": "Try to specify your intent clearly (e.g., record attendance, update, report, query)."
                }
        intent = parsed.get("intent")
        entities = parsed.get("entities", {})
        # Route to appropriate handler
        async with trace_step(
            "route_intent", step_type="plan",
            decision_summary="Route to appropriate handler based on intent",
            output_fn=lambda r: f"intent={intent}"
        ) as step:
            if intent == "record_attendance":
                try:
                    result = await self.domain_engine.record_attendance(
                        student_id=entities.get("student_id"),
                        date=entities.get("date"),
                        status=entities.get("status"),
                        user_id=user_id
                    )
                except Exception as e:
                    logger.error(f"Error in record_attendance: {e}")
                    result = {
                        "success": False,
                        "error": "ERR_INTERNAL",
                        "message": "Failed to record attendance.",
                        "fix_tip": "Try again later."
                    }
            elif intent == "update_attendance":
                try:
                    result = await self.domain_engine.update_attendance(
                        student_id=entities.get("student_id"),
                        date=entities.get("date"),
                        new_status=entities.get("status"),
                        user_id=user_id
                    )
                except Exception as e:
                    logger.error(f"Error in update_attendance: {e}")
                    result = {
                        "success": False,
                        "error": "ERR_INTERNAL",
                        "message": "Failed to update attendance.",
                        "fix_tip": "Try again later."
                    }
            elif intent == "generate_report":
                try:
                    result = await self.domain_engine.generate_report(
                        report_type=entities.get("report_type", "student"),
                        filters=entities.get("filters", {}),
                        user_id=user_id
                    )
                except Exception as e:
                    logger.error(f"Error in generate_report: {e}")
                    result = {
                        "success": False,
                        "error": "ERR_INTERNAL",
                        "message": "Failed to generate report.",
                        "fix_tip": "Try again later."
                    }
            elif intent == "query_attendance_status":
                try:
                    result = await self.domain_engine.query_attendance_status(
                        student_id=entities.get("student_id"),
                        class_id=entities.get("class_id"),
                        date=entities.get("date"),
                        user_id=user_id
                    )
                except Exception as e:
                    logger.error(f"Error in query_attendance_status: {e}")
                    result = {
                        "success": False,
                        "error": "ERR_INTERNAL",
                        "message": "Failed to query attendance.",
                        "fix_tip": "Try again later."
                    }
            else:
                result = {
                    "success": False,
                    "error": "ERR_UNKNOWN_INTENT",
                    "message": "Could not determine your intent.",
                    "fix_tip": "Try to specify your intent clearly."
                }
            step.capture(result)
        return result

# --- Presentation Layer: User Interface Handler ---

class UserInterfaceHandler:
    """Handles incoming requests, formats responses, manages user sessions."""
    def __init__(self, app_controller: ApplicationController, auth_manager: AuthenticationManager):
        self.app_controller = app_controller
        self.auth_manager = auth_manager

    async def handle_input(self, input_data: AttendanceInput) -> Dict[str, Any]:
        async with trace_step(
            "handle_input", step_type="process",
            decision_summary="Handle user input and route to controller",
            output_fn=lambda r: f"success={r.get('success')}"
        ) as step:
            try:
                result = await self.app_controller.process_input(
                    input_text=input_data.input_text,
                    user_id=input_data.user_id,
                    communication_channel=input_data.communication_channel
                )
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"Error in handle_input: {e}")
                return {
                    "success": False,
                    "error": "ERR_INTERNAL",
                    "message": "Internal error occurred.",
                    "fix_tip": "Try again later."
                }

# --- Main Agent Class ---

class BaseAgent:
    """Abstract base agent class."""
    pass

class AttendanceAgent(BaseAgent):
    """Student Attendance Management Agent."""
    def __init__(self):
        # Compose all components
        self.attendance_api = AttendanceAPIClient()
        self.audit_logger = AuditLogger()
        self.auth_manager = AuthenticationManager()
        self.domain_engine = DomainLogicEngine(self.attendance_api, self.audit_logger, self.auth_manager)
        self.app_controller = ApplicationController(self.domain_engine, self.auth_manager)
        self.ui_handler = UserInterfaceHandler(self.app_controller, self.auth_manager)

    @trace_agent(agent_name='Student Attendance Management Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_input(self, input_text: str, user_id: str, communication_channel: str = "chat") -> Dict[str, Any]:
        async with trace_step(
            "agent_process_input", step_type="final",
            decision_summary="Agent entrypoint for processing input",
            output_fn=lambda r: f"success={r.get('success')}"
        ) as step:
            result = await self.ui_handler.handle_input(
                AttendanceInput(
                    input_text=input_text,
                    user_id=user_id,
                    communication_channel=communication_channel
                )
            )
            step.capture(result)
            return result

    async def record_attendance(self, student_id: str, date: str, status: str, user_id: str) -> Dict[str, Any]:
        return await self.domain_engine.record_attendance(student_id, date, status, user_id)

    async def update_attendance(self, student_id: str, date: str, new_status: str, user_id: str) -> Dict[str, Any]:
        return await self.domain_engine.update_attendance(student_id, date, new_status, user_id)

    async def generate_report(self, report_type: str, filters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        return await self.domain_engine.generate_report(report_type, filters, user_id)

    @trace_agent(agent_name='Student Attendance Management Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def query_attendance_status(self, student_id: Optional[str], class_id: Optional[str], date: Optional[str], user_id: str) -> Dict[str, Any]:
        return await self.domain_engine.query_attendance_status(student_id, class_id, date, user_id)

    def validate_user(self, user_id: str, required_role: str) -> bool:
        return self.auth_manager.validate_user(user_id, required_role)

    def log_audit_entry(self, record_id: str, user_id: str, action: str, timestamp: str) -> bool:
        return self.audit_logger.log_audit_entry(record_id, user_id, action, timestamp)

# --- FastAPI App and Endpoints ---

app = FastAPI(
    title="Student Attendance Management Agent",
    description="API for managing student attendance with LLM-powered NLU and compliance.",
    version="1.0.0"
)

# CORS for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

agent = AttendanceAgent()

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "ERR_INVALID_INPUT",
            "message": "Input validation failed.",
            "fix_tip": "Check your input fields for correctness.",
            "details": exc.errors()
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": "ERR_HTTP",
            "message": exc.detail,
            "fix_tip": "Check your request and try again."
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "ERR_INTERNAL",
            "message": "Internal server error.",
            "fix_tip": "Try again later."
        }
    )

@app.post("/agent/process_input")
@with_content_safety(config=GUARDRAILS_CONFIG)
async def process_input_endpoint(input_data: AttendanceInput):
    """
    Main endpoint for processing user input (natural language).
    """
    try:
        result = await agent.process_input(
            input_text=input_data.input_text,
            user_id=input_data.user_id,
            communication_channel=input_data.communication_channel
        )
        return result
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in process_input_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal error.")

@app.post("/agent/record_attendance")
async def record_attendance_endpoint(data: AttendanceRecordInput):
    """
    Endpoint to record attendance for a student.
    """
    try:
        result = await agent.record_attendance(
            student_id=data.student_id,
            date=data.date,
            status=data.status,
            user_id=data.user_id
        )
        return result
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in record_attendance_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal error.")

@app.post("/agent/update_attendance")
async def update_attendance_endpoint(data: AttendanceUpdateInput):
    """
    Endpoint to update attendance for a student.
    """
    try:
        result = await agent.update_attendance(
            student_id=data.student_id,
            date=data.date,
            new_status=data.new_status,
            user_id=data.user_id
        )
        return result
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in update_attendance_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal error.")

@app.post("/agent/generate_report")
async def generate_report_endpoint(data: ReportRequestInput):
    """
    Endpoint to generate attendance report.
    """
    try:
        result = await agent.generate_report(
            report_type=data.report_type,
            filters=data.filters,
            user_id=data.user_id
        )
        return result
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in generate_report_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal error.")

@app.post("/agent/query_attendance_status")
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_attendance_status_endpoint(data: AttendanceQueryInput):
    """
    Endpoint to query attendance status.
    """
    try:
        result = await agent.query_attendance_status(
            student_id=data.student_id,
            class_id=data.class_id,
            date=data.date,
            user_id=data.user_id
        )
        return result
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in query_attendance_status_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal error.")

# --- JSON Error Handling for Malformed Requests ---

@app.middleware("http")
@with_content_safety(config=GUARDRAILS_CONFIG)
async def catch_malformed_json(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            await request.json()
        except Exception as e:
            logger.warning(f"Malformed JSON: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "ERR_MALFORMED_JSON",
                    "message": "Malformed JSON in request body.",
                    "fix_tip": "Check for missing quotes, commas, or brackets. Ensure your JSON is valid.",
                    "details": str(e)
                }
            )
    response = await call_next(request)
    return response

# --- Main Entrypoint ---



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting Student Attendance Management Agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())