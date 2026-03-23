"""Runtime compatibility fixes loaded automatically by Python.

This module is imported automatically by Python when present on sys.path.
It patches Streamlit calls so older local Streamlit installs do not crash on
newer keyword styles like use_container_width=True.
"""

from __future__ import annotations


def _patch_streamlit() -> None:
    try:
        import streamlit as st  # type: ignore
        from streamlit.delta_generator import DeltaGenerator  # type: ignore
    except Exception:
        return

    def _normalize_kwargs_for_layout(kwargs: dict, *, drop_use_container_width_on_retry: bool = False) -> dict:
        new_kwargs = dict(kwargs)
        if new_kwargs.get("width") == "stretch":
            new_kwargs.pop("width", None)
            new_kwargs.setdefault("use_container_width", True)
        if drop_use_container_width_on_retry:
            new_kwargs.pop("use_container_width", None)
            new_kwargs.pop("width", None)
        return new_kwargs

    def _wrap_method(method_name: str) -> None:
        original = getattr(DeltaGenerator, method_name, None)
        if original is None:
            return

        def wrapped(self, *args, **kwargs):
            normalized = _normalize_kwargs_for_layout(kwargs)
            try:
                return original(self, *args, **normalized)
            except TypeError:
                fallback = _normalize_kwargs_for_layout(kwargs, drop_use_container_width_on_retry=True)
                return original(self, *args, **fallback)

        setattr(DeltaGenerator, method_name, wrapped)

    for _name in ["dataframe", "plotly_chart", "link_button"]:
        _wrap_method(_name)

    # Refresh the top-level helpers after patching DeltaGenerator methods.
    try:
        st.dataframe = st._main.dataframe  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        st.plotly_chart = st._main.plotly_chart  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        st.link_button = st._main.link_button  # type: ignore[attr-defined]
    except Exception:
        pass


_patch_streamlit()

