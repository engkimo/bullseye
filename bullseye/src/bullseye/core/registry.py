from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class ComponentFactory:
    name: str
    factory: Callable[..., object]


class Registry:
    """Simple pluggable registry for pipeline components.

    Components: detector, recognizer, layout, table, reading_order, exporter, llm
    """

    def __init__(self) -> None:
        self._components: Dict[str, Dict[str, ComponentFactory]] = {}

    def register(self, component: str, name: str, factory: Callable[..., object]) -> None:
        comp = self._components.setdefault(component, {})
        if name in comp:
            raise ValueError(f"{component}:{name} already registered")
        comp[name] = ComponentFactory(name=name, factory=factory)

    def create(self, component: str, name: str, **kwargs) -> object:
        comp = self._components.get(component, {})
        if name not in comp:
            raise KeyError(f"Unknown {component}:{name}")
        return comp[name].factory(**kwargs)

    def has(self, component: str, name: str) -> bool:
        return name in self._components.get(component, {})

    def list(self, component: Optional[str] = None) -> Dict[str, Dict[str, ComponentFactory]]:
        if component is None:
            return self._components
        return self._components.get(component, {})

