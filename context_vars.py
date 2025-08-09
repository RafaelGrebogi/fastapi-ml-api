from contextvars import ContextVar
from typing import Optional

current_user_id: ContextVar[Optional[int]] = ContextVar("user_id", default=None)
current_service_id: ContextVar[Optional[int]] = ContextVar("service_id", default=None)
current_result_id: ContextVar[Optional[int]] = ContextVar("result_id", default=None)
current_DeviceId: ContextVar[Optional[int]] = ContextVar("DeviceId", default=None)
current_DeviceSerial: ContextVar[Optional[str]] = ContextVar("DeviceSerial", default=None)
current_SessionToken: ContextVar[Optional[str]] = ContextVar("SessionToken", default=None)
