@echo off
echo ====================================================
echo Starting Exam Anomaly Detection System (NATIVE MODE)
echo ====================================================

echo [1/3] Migrating local SQLite database...
cd dashboard
python manage.py migrate
cd ..

echo [2/3] Starting FastAPI Backend on port 8000...
start "FastAPI Backend" cmd /k "cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

echo [3/3] Starting Django Dashboard on port 8001...
start "Django Dashboard" cmd /k "cd dashboard && python manage.py runserver 0.0.0.0:8001"

echo ====================================================
echo All services started!
echo - Dashboard UI: http://localhost:8001
echo - API Backend: http://localhost:8000
echo ====================================================
