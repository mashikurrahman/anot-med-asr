#!/usr/bin/env python3
"""Tests for the API-key boundary, including the Deepgram-style Authorization header."""

import asyncio
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.security.privacy import verify_auth_boundary


def _check(x_api_key=None, authorization=None, expected="secret"):
    with patch("app.security.privacy.settings") as s:
        s.API_KEY = expected
        return asyncio.run(verify_auth_boundary(api_key=x_api_key, authorization=authorization))


class TestAuthBoundary(unittest.TestCase):
    def test_x_api_key_accepted(self):
        self.assertIsNone(_check(x_api_key="secret"))

    def test_deepgram_token_header_accepted(self):
        self.assertIsNone(_check(authorization="Token secret"))

    def test_bearer_header_accepted(self):
        self.assertIsNone(_check(authorization="Bearer secret"))

    def test_missing_or_wrong_key_rejected(self):
        for kw in ({}, {"x_api_key": "nope"}, {"authorization": "Token nope"}, {"authorization": "secret"}):
            with self.assertRaises(HTTPException) as cm:
                _check(**kw)
            self.assertEqual(cm.exception.status_code, 401)

    def test_no_key_configured_allows_all(self):
        self.assertIsNone(_check(expected=None))


if __name__ == "__main__":
    unittest.main()
