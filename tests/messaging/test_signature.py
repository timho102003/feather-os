"""Tests for LINE and WhatsApp webhook signature validation.

Signature validation is the only thing standing between Feather and a
forged webhook event, so a positive AND a negative test for each
algorithm is mandatory.
"""

from __future__ import annotations

import base64
import hashlib
import hmac

from feather.messaging.adapters.line import _verify_signature as line_verify
from feather.messaging.adapters.whatsapp import (
    _verify_signature as wa_verify,
)


def test_line_signature_accepts_correctly_signed_body() -> None:
    secret = "channel-secret"
    body = b'{"events":[]}'
    digest = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).digest()
    valid = base64.b64encode(digest).decode("ascii")

    assert line_verify(secret, body, valid) is True


def test_line_signature_rejects_wrong_secret() -> None:
    body = b'{"events":[]}'
    digest = hmac.new(b"different", body, hashlib.sha256).digest()
    wrong = base64.b64encode(digest).decode("ascii")

    assert line_verify("real-secret", body, wrong) is False


def test_line_signature_rejects_modified_body() -> None:
    secret = "channel-secret"
    digest = hmac.new(
        secret.encode("utf-8"), b'{"events":[]}', hashlib.sha256
    ).digest()
    valid = base64.b64encode(digest).decode("ascii")

    assert line_verify(secret, b'{"events":[{"type":"message"}]}', valid) is False


def test_line_signature_rejects_empty_header() -> None:
    assert line_verify("secret", b"body", "") is False


def test_whatsapp_signature_accepts_correct_hex_signature() -> None:
    secret = "app-secret"
    body = b'{"object":"whatsapp_business_account"}'
    digest = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    valid = f"sha256={digest}"

    assert wa_verify(secret, body, valid) is True


def test_whatsapp_signature_rejects_missing_prefix() -> None:
    secret = "app-secret"
    body = b"data"
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    # Missing the "sha256=" prefix.
    assert wa_verify(secret, body, digest) is False


def test_whatsapp_signature_rejects_wrong_secret() -> None:
    body = b"data"
    digest = hmac.new(b"wrong", body, hashlib.sha256).hexdigest()

    assert wa_verify("right", body, f"sha256={digest}") is False


def test_whatsapp_signature_rejects_empty_header() -> None:
    assert wa_verify("secret", b"body", "") is False


def test_whatsapp_signature_uses_constant_time_compare() -> None:
    """Compare-digest must reject signatures that differ only at the end.

    Cheap regression guard against accidentally swapping ``hmac.compare_digest``
    for a string ``==`` (which short-circuits on first mismatch and leaks
    timing info).
    """

    secret = "app-secret"
    body = b"x"
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    near_miss = "sha256=" + digest[:-1] + ("0" if digest[-1] != "0" else "1")

    assert wa_verify(secret, body, near_miss) is False
