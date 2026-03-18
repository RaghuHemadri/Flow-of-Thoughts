#!/bin/bash
# Usage: ./hpc_connect.sh <compute_node> [username]
#        ./hpc_connect.sh kill

COMPUTE_NODE=${1:?"Error: compute node required. Usage: $0 <compute_node> [username] | kill"}
USER=${2:-"rh3884"}
LOGIN_HOST="login.torch.hpc.nyu.edu"
PORT=60869
PID_FILE="/tmp/hpc_tunnel.pid"

if [ "$1" == "kill" ]; then
  if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    echo "==> Killing tunnel (PID: $PID)..."
    kill $PID
    rm $PID_FILE
    echo "==> Tunnel closed."
  else
    echo "==> No active tunnel found."
  fi
  exit 0
fi

echo "==> Starting tunnel: $USER@$LOGIN_HOST -> $COMPUTE_NODE on port $PORT..."

ssh -fN \
  -R ${PORT}:localhost:${PORT} \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  ${USER}@${LOGIN_HOST} \
  ssh -N -R ${PORT}:localhost:${PORT} ${COMPUTE_NODE} &

echo $! > $PID_FILE
echo "==> Tunnel established (PID: $!). Run '$0 kill' to close."