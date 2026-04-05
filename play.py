#!/usr/bin/env python3
"""
Summative rubric entry point: run the trained agent with visualization.

Delegates to the same loader as main.py. Examples:

    python play.py --algo dqn
    python play.py --algo ppo --episodes 3 --render human
"""

from __future__ import annotations

import main as _main

if __name__ == "__main__":
    _main.main()
