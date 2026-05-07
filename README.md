# pogema-agv

AGV-oriented fork of Pogema for four-action MAPF environments.

The Python import package remains `pogema` for compatibility, but the distribution
name is `pogema-agv`. The environment is oriented-action only:

- `0`: Forward
- `1`: TurnLeft
- `2`: TurnRight
- `3`: Wait

This package is derived from the local vendored Pogema used by `distill-lagat-agv`
and keeps the upstream Pogema package layout where practical.
