# bioamla.protocols

The interfaces bioamla accepts from callers — progress and cancel hooks, and audio
models — as `typing.Protocol`s. Implement them structurally; nothing needs to
inherit from them. The module imports only the standard library.

::: bioamla.protocols
    options:
      show_root_heading: false
