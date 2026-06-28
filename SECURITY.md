# Security Policy

## Supported versions

harken is pre-1.0; security fixes land on the latest released minor version.

| Version | Supported |
|---------|-----------|
| 0.3.x   | ✅        |
| < 0.3   | ❌        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, use
GitHub's private vulnerability reporting ("Report a vulnerability" under the
repository's Security tab), or contact the maintainer directly.

Include a description, reproduction steps, and the affected version. You can
expect an acknowledgement within a few days and a fix or mitigation plan once
the report is confirmed.

## Scope

harken loads audio files and, optionally, third-party model weights from the
Hugging Face hub. Treat untrusted audio and checkpoints with care: only load
weights from sources you trust, and run untrusted inputs in a sandbox.
