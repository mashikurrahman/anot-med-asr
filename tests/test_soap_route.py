#!/usr/bin/env python3
"""Tests for the /api/v1/soap route (SoapService is faked; no network, no models)."""

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.services.soap_service import SoapGenerationError

NOTE = {"subjective": "- s", "objective": "- o", "assessment": "- a", "plan": "- p", "flags": ["check dose"],
        "model": "test-model"}


def _client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


class TestSoapRoute(unittest.TestCase):
    def test_success(self):
        with patch.object(routes._soap, "generate", new=AsyncMock(return_value=NOTE)):
            r = _client().post("/api/v1/soap", json={"transcript": "Doctor: take prednisone."})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["plan"], "- p")
        self.assertEqual(r.json()["flags"], ["check dose"])

    def test_empty_transcript_is_422(self):
        r = _client().post("/api/v1/soap", json={"transcript": ""})
        self.assertEqual(r.status_code, 422)

    def test_service_errors_map_to_status_and_safe_detail(self):
        err = SoapGenerationError("SOAP generation is not configured on this server.", 503)
        with patch.object(routes._soap, "generate", new=AsyncMock(side_effect=err)):
            r = _client().post("/api/v1/soap", json={"transcript": "hello"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("not configured", r.json()["detail"])

    def test_api_key_required_when_configured(self):
        with patch("app.security.privacy.settings") as s:
            s.API_KEY = "secret"
            r = _client().post("/api/v1/soap", json={"transcript": "hello"})
        self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()
