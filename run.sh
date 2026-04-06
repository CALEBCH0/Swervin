
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

sleep 5
python src/sas/gui_frontend.py