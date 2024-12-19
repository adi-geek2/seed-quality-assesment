@echo off
REM Activate the virtual environment
call venv\Scripts\activate

REM Run the Python script with the arguments
python %*
