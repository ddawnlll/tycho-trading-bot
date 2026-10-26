"""Compatibility entry point for the single Nautilus paper bot."""
import runpy

if __name__ == "__main__":
    runpy.run_module("scripts.nautilus_live", run_name="__main__")
