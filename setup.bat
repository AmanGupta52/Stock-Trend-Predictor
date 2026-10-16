@echo off
echo Setting up Stock Trend Predictor...
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete. To start the dashboard:
echo   venv\Scripts\activate
echo   python app.py
echo then open http://127.0.0.1:5000
pause
