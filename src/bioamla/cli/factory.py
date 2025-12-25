"""Service factory for dependency injection."""

from typing import Optional, Type, TypeVar

from src.bioamla.repository.local import LocalFileRepository
from src.bioamla.repository.protocol import FileRepositoryProtocol

T = TypeVar("T")


class ServiceFactory:
    """Factory for creating services with dependency injection.

    Manages repository instances and service instantiation to ensure
    proper dependency injection throughout the CLI.
    """

    def __init__(
        self,
        file_repository: Optional[FileRepositoryProtocol] = None,
    ) -> None:
        """Initialize the factory with a repository instance.

        Args:
            file_repository: Optional custom repository. Defaults to LocalFileRepository.
        """
        self.file_repository = (
            file_repository or LocalFileRepository()
        )

    def create_service(self, service_class: Type[T]) -> T:
        """Create a service instance with dependencies injected.

        Args:
            service_class: The service class to instantiate

        Returns:
            An instance of the service with dependencies injected
        """
        return service_class(self.file_repository)

    def set_repository(
        self, file_repository: FileRepositoryProtocol
    ) -> None:
        """Set a custom repository instance.

        Useful for testing with mock repositories.

        Args:
            file_repository: The repository to use
        """
        self.file_repository = file_repository
