"""
Signal Registry - Plugin system for signal registration and discovery.

Enables adding new signals without modifying core code through decorator-based
registration.

Example:
    @SignalRegistry.register("my_signal")
    class MySignal(Signal):
        ...

    # Later, retrieve and instantiate
    signal_cls = SignalRegistry.get("my_signal")
    signal = signal_cls(lookback=20)
"""

from typing import Any, Dict, List, Optional, Type

from .base import Signal


class SignalRegistryError(Exception):
    """Exception raised for signal registry errors."""

    pass


class SignalRegistry:
    """
    Central registry for all available signals.

    Provides a plugin architecture allowing signals to be registered
    via decorator and discovered at runtime.

    Class Attributes:
        _signals: Dictionary mapping signal names to signal classes
        _metadata: Dictionary mapping signal names to metadata

    Example:
        # Register a signal
        @SignalRegistry.register("momentum_10d")
        class Momentum10DSignal(Signal):
            ...

        # Get all registered signals
        all_signals = SignalRegistry.list_all()

        # Instantiate a signal by name
        signal = SignalRegistry.create("momentum_10d", lookback=10)
    """

    _signals: Dict[str, Type[Signal]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        category: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Decorator to register a signal class.

        Args:
            name: Unique identifier for the signal
            category: Signal category (e.g., 'momentum', 'mean_reversion')
            description: Human-readable description
            tags: List of tags for filtering

        Returns:
            Decorator function

        Raises:
            SignalRegistryError: If name is already registered

        Example:
            @SignalRegistry.register(
                "rsi_oversold",
                category="mean_reversion",
                description="RSI-based oversold/overbought signal",
                tags=["oscillator", "reversal"]
            )
            class RSIOversoldSignal(Signal):
                ...
        """

        def decorator(signal_cls: Type[Signal]) -> Type[Signal]:
            if name in cls._signals:
                raise SignalRegistryError(
                    f"Signal '{name}' is already registered. "
                    f"Currently registered: {cls._signals[name].__name__}"
                )

            if not issubclass(signal_cls, Signal):
                raise SignalRegistryError(
                    f"Class {signal_cls.__name__} must be a subclass of Signal"
                )

            cls._signals[name] = signal_cls
            cls._metadata[name] = {
                "category": category,
                "description": description or signal_cls.__doc__,
                "tags": tags or [],
                "class_name": signal_cls.__name__,
            }

            return signal_cls

        return decorator

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove a signal from the registry.

        Args:
            name: Signal name to remove

        Raises:
            SignalRegistryError: If signal is not registered
        """
        if name not in cls._signals:
            raise SignalRegistryError(f"Signal '{name}' is not registered")

        del cls._signals[name]
        del cls._metadata[name]

    @classmethod
    def get(cls, name: str) -> Type[Signal]:
        """
        Get a signal class by name.

        Args:
            name: Signal name

        Returns:
            Signal class

        Raises:
            SignalRegistryError: If signal is not found
        """
        if name not in cls._signals:
            available = ", ".join(cls._signals.keys())
            raise SignalRegistryError(
                f"Signal '{name}' not found. Available signals: {available}"
            )
        return cls._signals[name]

    @classmethod
    def create(cls, name: str, **params: Any) -> Signal:
        """
        Create a signal instance by name.

        Args:
            name: Signal name
            **params: Parameters to pass to signal constructor

        Returns:
            Signal instance

        Example:
            signal = SignalRegistry.create("momentum", lookback=20)
        """
        signal_cls = cls.get(name)
        return signal_cls(**params)

    @classmethod
    def list_all(cls) -> List[str]:
        """
        Get list of all registered signal names.

        Returns:
            List of signal names
        """
        return list(cls._signals.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """
        Get signals filtered by category.

        Args:
            category: Category to filter by

        Returns:
            List of signal names in the category
        """
        return [
            name
            for name, meta in cls._metadata.items()
            if meta.get("category") == category
        ]

    @classmethod
    def list_by_tag(cls, tag: str) -> List[str]:
        """
        Get signals filtered by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of signal names with the tag
        """
        return [
            name
            for name, meta in cls._metadata.items()
            if tag in meta.get("tags", [])
        ]

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a signal.

        Args:
            name: Signal name

        Returns:
            Metadata dictionary

        Raises:
            SignalRegistryError: If signal is not found
        """
        if name not in cls._metadata:
            raise SignalRegistryError(f"Signal '{name}' not found")
        return cls._metadata[name].copy()

    @classmethod
    def get_all_metadata(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all registered signals.

        Returns:
            Dictionary mapping signal names to metadata
        """
        return {name: meta.copy() for name, meta in cls._metadata.items()}

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered signals.

        Warning: Use only for testing purposes.
        """
        cls._signals.clear()
        cls._metadata.clear()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a signal is registered.

        Args:
            name: Signal name

        Returns:
            True if registered, False otherwise
        """
        return name in cls._signals

    @classmethod
    def categories(cls) -> List[str]:
        """
        Get list of all unique categories.

        Returns:
            List of category names
        """
        categories = set()
        for meta in cls._metadata.values():
            if meta.get("category"):
                categories.add(meta["category"])
        return sorted(categories)

    @classmethod
    def summary(cls) -> Dict[str, Any]:
        """
        Get summary of registered signals.

        Returns:
            Summary dictionary with counts and categories
        """
        return {
            "total_signals": len(cls._signals),
            "categories": cls.categories(),
            "signals_by_category": {
                cat: cls.list_by_category(cat) for cat in cls.categories()
            },
        }
