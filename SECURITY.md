# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |

## Reporting a Vulnerability

Please report security vulnerabilities by opening a **private** GitHub issue or emailing devneatharva@gmail.com.

Do not open a public issue for security vulnerabilities.

We will respond within 48 hours and aim to release a patch within 7 days.

## Security Practices

- All secrets must be set via environment variables (see `.env.example`)
- No credentials are committed to the repository
- SQL queries use parameterized statements via SQLAlchemy ORM
- Input is validated at all API boundaries using Pydantic
- Docker containers run as a non-root user
