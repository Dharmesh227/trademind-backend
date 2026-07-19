"""Push notification API endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from trademind.engines.notifications.service import notification_service

router = APIRouter(prefix="/notifications", tags=["Push Notifications"])


# ── Request / Response models ──────────────────────────────


class RegisterTokenRequest(BaseModel):
    token: str
    device_info: str = ""


class UnregisterTokenRequest(BaseModel):
    token: str


class SendNotificationRequest(BaseModel):
    title: str
    body: str
    topic: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────


@router.post("/register")
async def register_token(req: RegisterTokenRequest):
    """Register a device token for push notifications."""
    notification_service.add_token(req.token, req.device_info)
    return {
        "status": "registered",
        "total_tokens": len(notification_service.get_tokens()),
    }


@router.post("/unregister")
async def unregister_token(req: UnregisterTokenRequest):
    """Remove a device token from the registry."""
    removed = notification_service.remove_token(req.token)
    if not removed:
        raise HTTPException(status_code=404, detail="Token not found")
    return {
        "status": "unregistered",
        "total_tokens": len(notification_service.get_tokens()),
    }


@router.get("/tokens")
async def list_tokens():
    """List all registered device tokens."""
    return {
        "tokens": notification_service.get_tokens(),
        "total": len(notification_service.get_tokens()),
    }


@router.post("/test")
async def send_test_notification():
    """Send a test notification to the first registered device."""
    tokens = notification_service.get_tokens()
    if not tokens:
        raise HTTPException(
            status_code=404,
            detail="No device tokens registered. Register a device first.",
        )

    first_token = tokens[0]["token"]
    success = await notification_service.send_to_token(
        token=first_token,
        title="TradeMind AI",
        body="Test notification — push notifications are working!",
        data={"type": "test"},
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to send test notification. Check FCM configuration.",
        )

    return {"status": "sent", "token_prefix": first_token[:20] + "..."}


@router.post("/send")
async def send_notification(req: SendNotificationRequest):
    """Send a notification to a topic or all registered devices."""
    if req.topic:
        success = await notification_service.send_to_topic(
            topic=req.topic,
            title=req.title,
            body=req.body,
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to send topic notification")
        return {"status": "sent", "target": f"topic:{req.topic}"}

    results = await notification_service.send_to_all(
        title=req.title,
        body=req.body,
    )
    return {"status": "completed", **results}
