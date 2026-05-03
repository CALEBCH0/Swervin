PROFILE_FLAG=""
SESSION_NAME=""

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--profile)
            PROFILE_FLAG="--profile"
            shift # past argument
            ;;
        -h|--help)
            echo "Usage: $0 [-p|--profile] [session_name]"
            echo "  -p, --profile      Run with profiling enabled"
            echo "  session_name       Optional session name for recording (passed to --record)"
            exit 0
            ;;
        *)
            # If argument does not start with dash, treat as session name
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

# Add src directory to Python path so 'sas' module can be found
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

CMD="python src/sas/run_sas.py"

# Add flags as needed
if [ -n "$PROFILE_FLAG" ]; then
    CMD="$CMD $PROFILE_FLAG"
fi

if [ -n "$SESSION_NAME" ]; then
    CMD="$CMD --record $SESSION_NAME"
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
# Trap signals to ensure cleanup
trap cleanup SIGINT SIGTERM EXIT

echo "Server started. Press Ctrl+C to stop gracefully."
echo "Server listening on port 65432"

echo "Waiting for server..."
while ! ss -tln | grep -q ':65432'; do
    sleep 0.5
done
echo "Server ready."
QT_QPA_PLATFORM=wayland python src/sas/gui_frontend.py