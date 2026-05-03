PROFILE_FLAG=""
SESSION_NAME=""
CALIBRATE_BEV_FLAG=""

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--profile)
            PROFILE_FLAG="--profile"
            shift
            ;;
        --calibrate-bev)
            CALIBRATE_BEV_FLAG="--calibrate-bev"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-p|--profile] [--calibrate-bev] [session_name]"
            echo "  -p, --profile      Run with profiling enabled"
            echo "  --calibrate-bev    Run interactive BEV calibration and save homography, then exit"
            echo "  session_name       Optional session name for recording (passed to --record)"
            exit 0
            ;;
        *)
            if [[ -z "$SESSION_NAME" && "$1" != -* ]]; then
                SESSION_NAME="$1"
                shift
            else
                echo "Unknown option or extra argument: $1"
                echo "Use -h for help"
                exit 1
            fi
            ;;
    esac
done

cd '/mnt/c/Users/Caleb Cho/code/Swervin'
source .venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

CMD="python src/sas/run_sas.py"

if [ -n "$PROFILE_FLAG" ]; then
    CMD="$CMD $PROFILE_FLAG"
fi

if [ -n "$SESSION_NAME" ]; then
    CMD="$CMD --record $SESSION_NAME"
fi

if [ -n "$CALIBRATE_BEV_FLAG" ]; then
    CMD="$CMD $CALIBRATE_BEV_FLAG"
    echo "Running: $CMD"
    $CMD
    exit $?
fi

# Echo final command for debug
echo "Running: $CMD &"
$CMD &
SERVER_PID=$!

# Store PID for cleanup
echo "Server PID: $SERVER_PID"

# Setup signal handlers for graceful shutdown
cleanup() {
      echo "Shutting down server (PID: $SERVER_PID)..."
      if kill -0 $SERVER_PID 2>/dev/null; then
          kill -TERM $SERVER_PID 2>/dev/null
          sleep 2
          kill -KILL $SERVER_PID 2>/dev/null || true
      fi
      echo "Cleanup complete."
      exit 0
  }
trap cleanup SIGINT SIGTERM EXIT

echo "Server started. Press Ctrl+C to stop gracefully."
echo "Server listening on port 65432"

echo "Waiting for server..."
while ! ss -tln | grep -q ':65432'; do
    sleep 0.5
done
echo "Server ready."
LIBGL_ALWAYS_SOFTWARE=1 QT_QPA_PLATFORM=wayland python src/sas/gui_frontend.py
