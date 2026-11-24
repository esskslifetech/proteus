@echo off
set ROOT=C:\Users\flopt\AppData\Local\Temp\opencode\proteus
set VENV=%ROOT%\python\.venv
if not exist "%ROOT%\cpp\build-msvc" mkdir "%ROOT%\cpp\build-msvc"
cd /d "%ROOT%\cpp\build-msvc"
"C:\msys64\mingw64\bin\cmake.exe" .. -G "Visual Studio 18 2026" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DEigen3_DIR=C:/msys64/mingw64/share/eigen3/cmake ^
  -Dpybind11_DIR="%VENV%/Lib/site-packages/pybind11/share/cmake/pybind11" ^
  -DPYBIND11_FINDPYTHON=ON ^
  -DPython_EXECUTABLE="%VENV%/Scripts/python.exe" ^
  -DCMAKE_DISABLE_FIND_PACKAGE_Sundials=ON ^
  -DCMAKE_DISABLE_FIND_PACKAGE_HiGHS=ON
if errorlevel 1 exit /b 1
"C:\msys64\mingw64\bin\cmake.exe" --build . --config Release -j4
