import time
from typing import Any, Dict, Optional
from pydantic import BaseModel


class HTTPCode:
    """HTTP status codes"""
    SUCCESS = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    UNPROCESSABLE_ENTITY = 422
    INTERNAL_SERVER_ERROR = 500


class NetworkResponse:
    """Network response utility class"""
    
    def success_response(
        self,
        http_code: int,
        message: str,
        data: Any,
        resource: str,
        start_time: float
    ) -> Dict[str, Any]:
        """Create a success response"""
        duration = f"{(time.time() - start_time):.3f}s"
        
        return {
            "success": True,
            "message": message,
            "data": data,
            "resource": resource,
            "duration": duration
        }
    
    def error_response(
        self,
        http_code: int,
        message: str,
        resource: str,
        start_time: float,
        error_details: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create an error response"""
        duration = f"{(time.time() - start_time):.3f}s"
        
        response = {
            "success": False,
            "message": message,
            "resource": resource,
            "duration": duration
        }
        
        if error_details:
            response["error_details"] = error_details
            
        return response