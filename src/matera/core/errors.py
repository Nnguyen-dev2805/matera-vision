from dataclasses import dataclass


@dataclass
class ProfileValidationError(Exception):
    profile_path: str
    field_path: str
    error_code: str
    reason: str

    def __str__(self) -> str:
        return f"{self.profile_path}: {self.field_path}: [{self.error_code}] {self.reason}"
