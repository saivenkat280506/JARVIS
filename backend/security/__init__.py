"""JARVIS Security and Authorization Architecture."""
from security.approval import (
    ActionApprovalManager,
    ApprovalStatus,
    AuditRecord,
    PendingAction,
    SecurityApprovalError,
    SecurityAuditStorageError,
    classify_confirmation_intent,
    compute_arguments_hash,
    default_approval_manager,
    sanitize_arguments,
    sanitize_audit_text,
)

__all__ = [
    "ActionApprovalManager",
    "ApprovalStatus",
    "AuditRecord",
    "PendingAction",
    "SecurityApprovalError",
    "SecurityAuditStorageError",
    "classify_confirmation_intent",
    "compute_arguments_hash",
    "default_approval_manager",
    "sanitize_arguments",
    "sanitize_audit_text",
]
