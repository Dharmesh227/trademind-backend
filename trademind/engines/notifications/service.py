"""Push notification service using Firebase Cloud Messaging (FCM HTTP v1 API)."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import jwt
from loguru import logger


class NotificationService:
    """Send push notifications via Firebase Cloud Messaging."""

    FCM_URL = "https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"

    def __init__(
        self,
        service_account_key_path: str | None = None,
    ) -> None:
        self._tokens: List[Dict[str, Any]] = []
        self._service_account_key: dict | None = None
        self._project_id: str | None = None
        self._oauth_token: str | None = None
        self._oauth_token_expiry: float = 0.0

        key_path = service_account_key_path or os.getenv("FCM_SERVICE_ACCOUNT_KEY")
        self._project_id = os.getenv("FCM_PROJECT_ID")

        if key_path and Path(key_path).exists():
            try:
                with open(key_path, "r", encoding="utf-8") as f:
                    self._service_account_key = json.load(f)
                self._project_id = self._project_id or self._service_account_key.get("project_id")
                logger.info("FCM service account loaded from %s", key_path)
            except Exception as e:
                logger.warning("Failed to load FCM service account from %s: %s", key_path, e)
                self._service_account_key = None
        else:
            if key_path:
                logger.warning("FCM service account file not found: %s", key_path)
            logger.info("FCM not configured — notifications will be logged only (development mode)")

    @property
    def _is_configured(self) -> bool:
        return self._service_account_key is not None and self._project_id is not None

    def _generate_oauth_token(self) -> str | None:
        """Generate a short-lived OAuth2 token from the service account key."""
        if not self._service_account_key:
            return None

        now = time.time()
        if self._oauth_token and now < self._oauth_token_expiry - 60:
            return self._oauth_token

        try:
            private_key = self._service_account_key["private_key"]
            client_email = self._service_account_key["client_email"]

            now_int = int(now)
            payload = {
                "iss": client_email,
                "scope": "https://www.googleapis.com/auth/firebase.messaging",
                "aud": "https://oauth2.googleapis.com/token",
                "iat": now_int,
                "exp": now_int + 3600,
            }

            token = jwt.encode(payload, private_key, algorithm="RS256")

            token_data = jwt.decode(token, private_key, algorithms=["RS256"])
            self._oauth_token = token
            self._oauth_token_expiry = token_data["exp"]

            return token
        except Exception as e:
            logger.error("Failed to generate FCM OAuth token: %s", e)
            self._oauth_token = None
            self._oauth_token_expiry = 0.0
            return None

    async def _send_fcm_message(self, message: dict) -> bool:
        """Send a single FCM message payload."""
        if not self._is_configured:
            logger.info(
                "FCM notification (dev mode) — %s",
                json.dumps(message, indent=2),
            )
            return True

        oauth_token = self._generate_oauth_token()
        if not oauth_token:
            logger.error("Cannot send FCM notification — no valid OAuth token")
            return False

        url = self.FCM_URL.format(project_id=self._project_id)
        headers = {
            "Authorization": f"Bearer {oauth_token}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json={"message": message}, headers=headers)

            if resp.status_code == 200:
                logger.info("FCM notification sent successfully")
                return True
            else:
                logger.error(
                    "FCM API error %d: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return False
        except httpx.TimeoutException:
            logger.error("FCM request timed out")
            return False
        except Exception as e:
            logger.error("FCM send failed: %s", e)
            return False

    def _build_message(
        self,
        title: str,
        body: str,
        data: dict | None = None,
        token: str | None = None,
        topic: str | None = None,
    ) -> dict:
        """Build an FCM message payload."""
        notification = {
            "title": title,
            "body": body,
        }

        message: dict[str, Any] = {"notification": notification}

        if data:
            message["data"] = {k: str(v) for k, v in data.items()}

        if token:
            message["token"] = token
        elif topic:
            message["topic"] = topic

        return message

    async def send_to_token(
        self,
        token: str,
        title: str,
        body: str,
        data: dict | None = None,
    ) -> bool:
        """Send a push notification to a specific device token."""
        message = self._build_message(title=title, body=body, data=data, token=token)
        return await self._send_fcm_message(message)

    async def send_to_topic(
        self,
        topic: str,
        title: str,
        body: str,
        data: dict | None = None,
    ) -> bool:
        """Send a push notification to all devices subscribed to a topic."""
        message = self._build_message(title=title, body=body, data=data, topic=topic)
        return await self._send_fcm_message(message)

    async def send_to_all(
        self,
        title: str,
        body: str,
        data: dict | None = None,
    ) -> dict:
        """Send a push notification to all registered devices.

        Returns a dict with counts of successful and failed sends.
        """
        results = {"sent": 0, "failed": 0, "total": len(self._tokens)}

        for entry in self._tokens:
            token = entry.get("token", "")
            if not token:
                results["failed"] += 1
                continue

            success = await self.send_to_token(token, title, body, data)
            if success:
                results["sent"] += 1
            else:
                results["failed"] += 1

        logger.info(
            "Broadcast notification: %d sent, %d failed, %d total",
            results["sent"],
            results["failed"],
            results["total"],
        )
        return results

    def add_token(self, token: str, device_info: str = "") -> None:
        """Register a device token for push notifications."""
        if any(entry.get("token") == token for entry in self._tokens):
            logger.info("Token already registered, updating device info")
            for entry in self._tokens:
                if entry.get("token") == token:
                    entry["device_info"] = device_info
                    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
                    return

        self._tokens.append(
            {
                "token": token,
                "device_info": device_info,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info("Device token registered (device=%s, total=%d)", device_info, len(self._tokens))

    def remove_token(self, token: str) -> bool:
        """Remove a device token from the registry."""
        before = len(self._tokens)
        self._tokens = [entry for entry in self._tokens if entry.get("token") != token]
        removed = len(self._tokens) < before
        if removed:
            logger.info("Device token removed (total=%d)", len(self._tokens))
        else:
            logger.warning("Token not found for removal")
        return removed

    def get_tokens(self) -> list:
        """Return a copy of all registered device tokens."""
        return list(self._tokens)


# Module-level singleton
notification_service = NotificationService()
