
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class ToDictMixin:
    """
    Mixin that adds to_dict() method to dataclasses.

    Handles nested dataclasses, lists, and common types automatically.
    Override _to_dict_extra() to add custom serialization logic.

    Example:
        @dataclass
        class MyResult(ToDictMixin):
            name: str
            count: int

        result = MyResult(name="test", count=5)
        data = result.to_dict()  # {"name": "test", "count": 5}
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary, handling nested structures."""
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} is not a dataclass")

        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            result[f.name] = self._serialize_value(value)

        # Allow subclasses to add extra fields
        extra = self._to_dict_extra()
        if extra:
            result.update(extra)

        return result

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value for dict output."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if is_dataclass(value):
            return asdict(value)
        # Fallback for other types
        return str(value)

    def _to_dict_extra(self) -> Optional[Dict[str, Any]]:
        """Override to add extra fields to dict output."""
        return None
