@echo off
REM Get the directory of the currently executed batch file
set SCRIPT_DIR=%~dp0

CALL conda init
REM Activate the Conda environment
CALL conda activate VITS

REM Run the Python script
python "%SCRIPT_DIR%qt.py"

REM Deactivate the Conda environment (optional)
CALL conda deactivate