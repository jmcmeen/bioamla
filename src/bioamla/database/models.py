# database/models.py
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class TimestampMixin(SQLModel):
    """Mixin for created/updated timestamps."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)


# Define your domain models here, e.g.:
#
# class ItemBase(SQLModel):
#     """Shared item properties."""
#     name: str = Field(index=True)
#     description: Optional[str] = None
#
#
# class Item(ItemBase, TimestampMixin, table=True):
#     """Item database model."""
#     __tablename__ = "items"
#
#     id: Optional[int] = Field(default=None, primary_key=True)
#
#
# class ItemCreate(ItemBase):
#     """Schema for creating an item."""
#     pass
#
#
# class ItemRead(ItemBase):
#     """Schema for reading an item."""
#     id: int
#     created_at: datetime
#
#
# class ItemUpdate(SQLModel):
#     """Schema for updating an item (all fields optional)."""
#     name: Optional[str] = None
#     description: Optional[str] = None
