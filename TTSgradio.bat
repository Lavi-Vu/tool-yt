REM Get the directory of the currently executed batch file
set SCRIPT_DIR=%~dp0

CALL conda init
REM Activate the Conda environment
CALL conda activate VITS

REM Run the Python script
python "%SCRIPT_DIR%TTSgradio.py --model_dir OUTPUT_MODEL/G_latest.pth --config_dir OUTPUT_MODEL/config.json"

REM Deactivate the Conda environment (optional)
CALL conda deactivate