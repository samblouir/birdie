from __future__ import annotations

import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class OptionalBuildExt(_build_ext):
	"""
	Build C/C++ extensions if possible, but do not fail installation if a compiler
	(or Python headers) are missing.
	"""

	def run(self):  # noqa: D401
		try:
			super().run()
		except Exception as exc:  # noqa: BLE001
			print(f"[birdie-rl] Warning: skipping optional C++ extensions (build_ext failed): {exc}", file=sys.stderr)

	def build_extension(self, ext):  # noqa: ANN001
		try:
			super().build_extension(ext)
		except Exception as exc:  # noqa: BLE001
			print(
				f"[birdie-rl] Warning: skipping optional extension {ext.name!r} (compile failed): {exc}",
				file=sys.stderr,
			)


def _cpp_compile_args() -> list[str]:
	if os.name == "nt":
		return ["/O2"]
	return ["-O3", "-std=c++11", "-pthread"]


def _cpp_link_args() -> list[str]:
	if os.name == "nt":
		return []
	return ["-pthread"]


ext_modules = []
if os.environ.get("BIRDIE_DISABLE_FAST_EXT", "0") != "1":
	ext_modules = [
		Extension(
			"birdie_rl.objectives._infilling_fast",
			sources=["birdie_rl/objectives/_infilling_fast.cpp"],
			language="c++",
			extra_compile_args=_cpp_compile_args(),
			extra_link_args=_cpp_link_args(),
		),
	]

# Metadata comes from `pyproject.toml` ([project]); we only define extensions here.
setup(cmdclass={"build_ext": OptionalBuildExt}, ext_modules=ext_modules)
