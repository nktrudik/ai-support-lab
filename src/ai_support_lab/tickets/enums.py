from enum import StrEnum


class Category(StrEnum):
    BILLING = "billing"
    AUTHENTICATION = "authentication"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    ACCOUNT = "account"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
