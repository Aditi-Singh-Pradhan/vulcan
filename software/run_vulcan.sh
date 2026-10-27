#!/bin/bash
export QT_QPA_PLATFORM=xcb
cd "$(dirname "$0")"
python3 -u -m core.vulcan2
