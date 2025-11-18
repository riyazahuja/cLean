#!/bin/bash
# Build script for cLean with BLAS compatibility
# This sets up the environment to use FlexiBLAS via local symlinks

export LIBRARY_PATH="/home/riyaza/cLean/.local/lib:/usr/lib64:$LIBRARY_PATH"
export C_INCLUDE_PATH="/home/riyaza/cLean/.local/include:/usr/include/flexiblas:$C_INCLUDE_PATH"
export LD_LIBRARY_PATH="/home/riyaza/cLean/.local/lib:/usr/lib64:$LD_LIBRARY_PATH"
export CFLAGS="-I/home/riyaza/cLean/.local/include -I/usr/include/flexiblas $CFLAGS"
export CPPFLAGS="-I/home/riyaza/cLean/.local/include -I/usr/include/flexiblas $CPPFLAGS"

# Run lake build with arguments
lake "$@"
