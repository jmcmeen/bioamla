# database/repository.py
from typing import Any, Generic, List, Optional, Type, TypeVar
from uuid import UUID

from sqlmodel import Session, SQLModel, select

ModelType = TypeVar("ModelType", bound=SQLModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=SQLModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=SQLModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Generic repository providing CRUD operations.

    Type Parameters:
        ModelType: The SQLModel table class
        CreateSchemaType: Pydantic model for creation
        UpdateSchemaType: Pydantic model for updates

    Usage:
        class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
            pass
    """

    def __init__(self, model: Type[ModelType], session: Session):
        self.model = model
        self.session = session

    # --- Create ---

    def create(self, obj_in: CreateSchemaType, **kwargs) -> ModelType:
        """
        Create a new record.

        Args:
            obj_in: Creation schema with input data
            **kwargs: Additional fields to set

        Returns:
            Created model instance
        """
        obj_data = obj_in.model_dump(exclude_unset=True)
        obj_data.update(kwargs)
        db_obj = self.model(**obj_data)
        self.session.add(db_obj)
        self.session.flush()
        self.session.refresh(db_obj)
        return db_obj

    def create_many(self, objects_in: List[CreateSchemaType]) -> List[ModelType]:
        """Create multiple records."""
        db_objects = [self.model(**obj.model_dump()) for obj in objects_in]
        self.session.add_all(db_objects)
        self.session.flush()
        for obj in db_objects:
            self.session.refresh(obj)
        return db_objects

    # --- Read ---

    def get(self, id: int | UUID) -> Optional[ModelType]:
        """Get a single record by ID."""
        return self.session.get(self.model, id)

    def get_or_raise(self, id: int | UUID) -> ModelType:
        """Get a single record by ID or raise ValueError."""
        obj = self.get(id)
        if obj is None:
            raise ValueError(f"{self.model.__name__} with id {id} not found")
        return obj

    def get_by_field(self, field: str, value: Any) -> Optional[ModelType]:
        """Get a single record by any field."""
        statement = select(self.model).where(getattr(self.model, field) == value)
        return self.session.exec(statement).first()

    def get_all(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        descending: bool = False,
    ) -> List[ModelType]:
        """
        Get all records with pagination.

        Args:
            skip: Number of records to skip
            limit: Maximum records to return
            order_by: Field name to order by
            descending: Sort descending if True
        """
        statement = select(self.model).offset(skip).limit(limit)

        if order_by and hasattr(self.model, order_by):
            order_field = getattr(self.model, order_by)
            statement = statement.order_by(order_field.desc() if descending else order_field)

        return list(self.session.exec(statement).all())

    def get_by_ids(self, ids: List[int | UUID]) -> List[ModelType]:
        """Get multiple records by their IDs."""
        statement = select(self.model).where(self.model.id.in_(ids))
        return list(self.session.exec(statement).all())

    def count(self) -> int:
        """Count total records."""
        from sqlalchemy import func

        statement = select(func.count()).select_from(self.model)
        return self.session.exec(statement).one()

    def exists(self, id: int | UUID) -> bool:
        """Check if a record exists."""
        return self.get(id) is not None

    # --- Update ---

    def update(self, db_obj: ModelType, obj_in: UpdateSchemaType | dict[str, Any]) -> ModelType:
        """
        Update an existing record.

        Args:
            db_obj: Existing database object
            obj_in: Update schema or dict with new values
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)

        if hasattr(db_obj, "updated_at"):
            from datetime import datetime

            db_obj.updated_at = datetime.utcnow()

        self.session.add(db_obj)
        self.session.flush()
        self.session.refresh(db_obj)
        return db_obj

    def update_by_id(
        self, id: int | UUID, obj_in: UpdateSchemaType | dict[str, Any]
    ) -> Optional[ModelType]:
        """Update a record by ID."""
        db_obj = self.get(id)
        if db_obj is None:
            return None
        return self.update(db_obj, obj_in)

    # --- Delete ---

    def delete(self, db_obj: ModelType) -> None:
        """Delete a record."""
        self.session.delete(db_obj)
        self.session.flush()

    def delete_by_id(self, id: int | UUID) -> bool:
        """Delete a record by ID. Returns True if deleted."""
        db_obj = self.get(id)
        if db_obj is None:
            return False
        self.delete(db_obj)
        return True

    def delete_many(self, ids: List[int | UUID]) -> int:
        """Delete multiple records. Returns count deleted."""
        objects = self.get_by_ids(ids)
        for obj in objects:
            self.session.delete(obj)
        self.session.flush()
        return len(objects)

    # --- Query Building ---

    def filter(self, **kwargs) -> List[ModelType]:
        """
        Filter records by field values.

        Example:
            repo.filter(is_active=True, name="John")
        """
        statement = select(self.model)
        for field, value in kwargs.items():
            if hasattr(self.model, field):
                statement = statement.where(getattr(self.model, field) == value)
        return list(self.session.exec(statement).all())
