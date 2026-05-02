#!/bin/bash
echo "[cc wrapper] invoked with: $@" >> /tmp/cc-linker-log.txt
exec /usr/bin/cc "$@"
