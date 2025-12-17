"""
pymetal.shader - Shader preprocessing utilities

This module provides utilities for preprocessing Metal shaders:
- Include directive resolution (#include "file.metal")
- Define macro substitution (#define NAME value)
- Template-based shader generation
- Shader caching and validation
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable


class ShaderPreprocessor:
    """
    Preprocessor for Metal shader source code.

    Supports:
    - #include "path" directive resolution
    - #define NAME value macro substitution
    - Custom macro functions
    - Include path management

    Example:
        preprocessor = ShaderPreprocessor()
        preprocessor.add_include_path("./shaders")
        preprocessor.define("BLOCK_SIZE", "256")

        source = preprocessor.process('''
            #include "common.metal"
            #define THREADS BLOCK_SIZE

            kernel void my_kernel(...) {
                // Uses BLOCK_SIZE = 256
            }
        ''')
    """

    # Regex patterns for preprocessing directives
    _INCLUDE_PATTERN = re.compile(r'^\s*#include\s*[<"]([^>"]+)[>"]\s*$', re.MULTILINE)
    _DEFINE_PATTERN = re.compile(r'^\s*#define\s+(\w+)(?:\s+(.*))?$', re.MULTILINE)
    _IFDEF_PATTERN = re.compile(r'^\s*#ifdef\s+(\w+)\s*$', re.MULTILINE)
    _IFNDEF_PATTERN = re.compile(r'^\s*#ifndef\s+(\w+)\s*$', re.MULTILINE)
    _ELSE_PATTERN = re.compile(r'^\s*#else\s*$', re.MULTILINE)
    _ENDIF_PATTERN = re.compile(r'^\s*#endif\s*$', re.MULTILINE)

    def __init__(self, include_paths: Optional[List[Union[str, Path]]] = None):
        """
        Initialize the shader preprocessor.

        Args:
            include_paths: List of directories to search for included files.
        """
        self._include_paths: List[Path] = []
        self._defines: Dict[str, str] = {}
        self._macro_functions: Dict[str, Callable[[str], str]] = {}
        self._include_cache: Dict[str, str] = {}
        self._processed_includes: set = set()

        if include_paths:
            for path in include_paths:
                self.add_include_path(path)

    def add_include_path(self, path: Union[str, Path]) -> "ShaderPreprocessor":
        """
        Add a directory to the include search path.

        Args:
            path: Directory path to add.

        Returns:
            Self for method chaining.
        """
        p = Path(path).resolve()
        if p not in self._include_paths:
            self._include_paths.append(p)
        return self

    def define(self, name: str, value: str = "1") -> "ShaderPreprocessor":
        """
        Define a macro for substitution.

        Args:
            name: Macro name (e.g., "BLOCK_SIZE").
            value: Macro value (e.g., "256").

        Returns:
            Self for method chaining.
        """
        self._defines[name] = str(value)
        return self

    def define_many(self, defines: Dict[str, str]) -> "ShaderPreprocessor":
        """
        Define multiple macros at once.

        Args:
            defines: Dictionary of name -> value mappings.

        Returns:
            Self for method chaining.
        """
        for name, value in defines.items():
            self.define(name, value)
        return self

    def undefine(self, name: str) -> "ShaderPreprocessor":
        """
        Remove a macro definition.

        Args:
            name: Macro name to remove.

        Returns:
            Self for method chaining.
        """
        self._defines.pop(name, None)
        return self

    def register_macro_function(
        self, name: str, func: Callable[[str], str]
    ) -> "ShaderPreprocessor":
        """
        Register a custom macro function.

        The function will be called when the macro is encountered,
        with the macro arguments as a string.

        Args:
            name: Macro name.
            func: Function that takes arguments string and returns replacement.

        Returns:
            Self for method chaining.

        Example:
            def repeat(args):
                count, text = args.split(',', 1)
                return text.strip() * int(count.strip())

            preprocessor.register_macro_function("REPEAT", repeat)
            # REPEAT(3, x) -> xxx
        """
        self._macro_functions[name] = func
        return self

    def _resolve_include(self, filename: str, current_file: Optional[Path] = None) -> str:
        """
        Resolve an include directive and return the file contents.

        Args:
            filename: The filename from the #include directive.
            current_file: The file containing the #include (for relative paths).

        Returns:
            Contents of the included file.

        Raises:
            FileNotFoundError: If the file cannot be found.
        """
        # Check cache first
        if filename in self._include_cache:
            return self._include_cache[filename]

        # Build search paths
        search_paths = list(self._include_paths)
        if current_file:
            search_paths.insert(0, current_file.parent)

        # Search for file
        for base_path in search_paths:
            full_path = base_path / filename
            if full_path.exists():
                content = full_path.read_text()
                self._include_cache[filename] = content
                return content

        raise FileNotFoundError(
            f"Cannot find included file '{filename}'. "
            f"Searched in: {[str(p) for p in search_paths]}"
        )

    def _process_includes(
        self, source: str, current_file: Optional[Path] = None, depth: int = 0
    ) -> str:
        """
        Process all #include directives in the source.

        Args:
            source: Shader source code.
            current_file: Current file path for relative includes.
            depth: Current recursion depth (to prevent infinite loops).

        Returns:
            Source with includes expanded.
        """
        if depth > 50:
            raise RecursionError("Include depth exceeded 50 - possible circular include")

        def replace_include(match: re.Match) -> str:
            filename = match.group(1)

            # Prevent duplicate includes (include guards)
            if filename in self._processed_includes:
                return f"// Already included: {filename}\n"
            self._processed_includes.add(filename)

            try:
                content = self._resolve_include(filename, current_file)
                # Recursively process includes in the included file
                include_path = None
                for base in self._include_paths:
                    if (base / filename).exists():
                        include_path = base / filename
                        break
                processed = self._process_includes(content, include_path, depth + 1)
                return f"// Begin include: {filename}\n{processed}\n// End include: {filename}\n"
            except FileNotFoundError as e:
                raise FileNotFoundError(str(e)) from None

        return self._INCLUDE_PATTERN.sub(replace_include, source)

    def _process_defines(self, source: str) -> str:
        """
        Process #define directives and substitute macros.

        Args:
            source: Shader source code.

        Returns:
            Source with defines processed and macros substituted.
        """
        # Collect defines from source
        local_defines = dict(self._defines)

        def collect_define(match: re.Match) -> str:
            name = match.group(1)
            value = match.group(2) if match.group(2) else "1"
            local_defines[name] = value.strip()
            return ""  # Remove the #define line

        # First pass: collect defines
        source = self._DEFINE_PATTERN.sub(collect_define, source)

        # Second pass: substitute macros (simple text replacement)
        # Sort by length (longest first) to avoid partial replacements
        for name in sorted(local_defines.keys(), key=len, reverse=True):
            value = local_defines[name]
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(name) + r'\b'
            source = re.sub(pattern, value, source)

        # Process macro functions
        for name, func in self._macro_functions.items():
            pattern = re.compile(rf'\b{re.escape(name)}\s*\(([^)]*)\)')
            source = pattern.sub(lambda m: func(m.group(1)), source)

        return source

    def _process_conditionals(self, source: str) -> str:
        """
        Process #ifdef/#ifndef/#else/#endif conditionals.

        Note: This is a simplified implementation that handles single-level
        conditionals. Nested conditionals are not fully supported.

        Args:
            source: Shader source code.

        Returns:
            Source with conditionals resolved.
        """
        lines = source.split('\n')
        result = []
        skip_until_endif = False
        skip_until_else = False
        in_conditional = False

        for line in lines:
            ifdef_match = self._IFDEF_PATTERN.match(line)
            ifndef_match = self._IFNDEF_PATTERN.match(line)
            else_match = self._ELSE_PATTERN.match(line)
            endif_match = self._ENDIF_PATTERN.match(line)

            if ifdef_match:
                name = ifdef_match.group(1)
                in_conditional = True
                if name in self._defines:
                    skip_until_else = False
                    skip_until_endif = False
                else:
                    skip_until_else = True
                continue
            elif ifndef_match:
                name = ifndef_match.group(1)
                in_conditional = True
                if name not in self._defines:
                    skip_until_else = False
                    skip_until_endif = False
                else:
                    skip_until_else = True
                continue
            elif else_match and in_conditional:
                if skip_until_else:
                    skip_until_else = False
                else:
                    skip_until_endif = True
                continue
            elif endif_match:
                in_conditional = False
                skip_until_else = False
                skip_until_endif = False
                continue

            if not skip_until_else and not skip_until_endif:
                result.append(line)

        return '\n'.join(result)

    def process(
        self,
        source: str,
        source_file: Optional[Union[str, Path]] = None,
        process_includes: bool = True,
        process_defines: bool = True,
        process_conditionals: bool = True,
    ) -> str:
        """
        Preprocess shader source code.

        Args:
            source: Shader source code to process.
            source_file: Optional path to source file (for relative includes).
            process_includes: Whether to process #include directives.
            process_defines: Whether to process #define and substitute macros.
            process_conditionals: Whether to process #ifdef/#ifndef conditionals.

        Returns:
            Preprocessed shader source code.
        """
        # Reset per-process state
        self._processed_includes.clear()

        current_file = Path(source_file).resolve() if source_file else None

        # Process in order: conditionals, includes, defines
        if process_conditionals:
            source = self._process_conditionals(source)

        if process_includes:
            source = self._process_includes(source, current_file)

        if process_defines:
            source = self._process_defines(source)

        return source

    def process_file(
        self,
        filepath: Union[str, Path],
        **kwargs,
    ) -> str:
        """
        Load and preprocess a shader file.

        Args:
            filepath: Path to the shader file.
            **kwargs: Additional arguments passed to process().

        Returns:
            Preprocessed shader source code.
        """
        path = Path(filepath)
        source = path.read_text()
        return self.process(source, source_file=path, **kwargs)

    def clear_cache(self) -> "ShaderPreprocessor":
        """
        Clear the include file cache.

        Returns:
            Self for method chaining.
        """
        self._include_cache.clear()
        return self


class ShaderTemplate:
    """
    Template-based shader generation.

    Uses Python string formatting to generate shaders with variable parameters.

    Example:
        template = ShaderTemplate('''
            #include <metal_stdlib>
            using namespace metal;

            kernel void {kernel_name}(
                device {dtype}* data [[buffer(0)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                data[idx] = data[idx] {operation};
            }}
        ''')

        # Generate a shader that doubles values
        source = template.render(
            kernel_name="double_values",
            dtype="float",
            operation="* 2.0"
        )
    """

    def __init__(self, template: str):
        """
        Initialize with a template string.

        Args:
            template: Shader template with {placeholder} markers.
                     Use {{ and }} for literal braces.
        """
        self._template = template
        self._defaults: Dict[str, str] = {}

    def set_default(self, name: str, value: str) -> "ShaderTemplate":
        """
        Set a default value for a placeholder.

        Args:
            name: Placeholder name.
            value: Default value.

        Returns:
            Self for method chaining.
        """
        self._defaults[name] = value
        return self

    def set_defaults(self, defaults: Dict[str, str]) -> "ShaderTemplate":
        """
        Set multiple default values.

        Args:
            defaults: Dictionary of placeholder -> value mappings.

        Returns:
            Self for method chaining.
        """
        self._defaults.update(defaults)
        return self

    def render(self, **kwargs) -> str:
        """
        Render the template with given parameters.

        Args:
            **kwargs: Values for placeholders.

        Returns:
            Rendered shader source code.
        """
        params = dict(self._defaults)
        params.update(kwargs)
        return self._template.format(**params)

    def render_with_preprocessor(
        self, preprocessor: ShaderPreprocessor, **kwargs
    ) -> str:
        """
        Render the template and then preprocess the result.

        Args:
            preprocessor: ShaderPreprocessor instance.
            **kwargs: Values for placeholders.

        Returns:
            Rendered and preprocessed shader source code.
        """
        rendered = self.render(**kwargs)
        return preprocessor.process(rendered)


def compute_shader_hash(source: str) -> str:
    """
    Compute a hash of shader source code.

    Useful for caching compiled pipelines.

    Args:
        source: Shader source code.

    Returns:
        SHA-256 hash as hex string.
    """
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


# Common shader snippets that can be included
COMMON_SNIPPETS = {
    "metal_stdlib": """
#include <metal_stdlib>
using namespace metal;
""",
    "float_constants": """
constant float PI = 3.14159265358979323846;
constant float TAU = 6.28318530717958647692;
constant float E = 2.71828182845904523536;
constant float SQRT2 = 1.41421356237309504880;
""",
    "int_constants": """
constant int INT_MAX = 2147483647;
constant int INT_MIN = -2147483648;
constant uint UINT_MAX = 4294967295u;
""",
}


def create_compute_kernel(
    name: str,
    body: str,
    buffers: List[tuple],
    threads_per_grid: bool = True,
    include_stdlib: bool = True,
) -> str:
    """
    Generate a compute kernel shader.

    Args:
        name: Kernel function name.
        body: Kernel body code (the code inside the function).
        buffers: List of (name, type, access) tuples.
                 access is "read" (device const), "write" (device), or "readwrite" (device).
        threads_per_grid: Whether to include thread_position_in_grid parameter.
        include_stdlib: Whether to include metal_stdlib header.

    Returns:
        Complete shader source code.

    Example:
        source = create_compute_kernel(
            name="vector_add",
            body="c[idx] = a[idx] + b[idx];",
            buffers=[
                ("a", "float", "read"),
                ("b", "float", "read"),
                ("c", "float", "write"),
            ]
        )
    """
    lines = []

    if include_stdlib:
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")

    # Build buffer parameters
    params = []
    for i, (buf_name, buf_type, access) in enumerate(buffers):
        if access == "read":
            params.append(f"device const {buf_type}* {buf_name} [[buffer({i})]]")
        else:
            params.append(f"device {buf_type}* {buf_name} [[buffer({i})]]")

    if threads_per_grid:
        params.append("uint idx [[thread_position_in_grid]]")

    # Build function
    param_str = ",\n                     ".join(params)
    lines.append(f"kernel void {name}({param_str}) {{")

    # Indent body
    for line in body.strip().split('\n'):
        lines.append(f"    {line}")

    lines.append("}")

    return '\n'.join(lines)


__all__ = [
    "ShaderPreprocessor",
    "ShaderTemplate",
    "compute_shader_hash",
    "create_compute_kernel",
    "COMMON_SNIPPETS",
]
