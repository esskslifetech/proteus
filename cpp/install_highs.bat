@echo off
set PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;C:\msys64\bin;%PATH%
C:\msys64\mingw64\bin\bash.exe -lc "export PATH=/mingw64/bin:/usr/bin:/bin:$PATH && which gcc && gcc -dumpmachine && pacman -S --noconfirm mingw-w64-x86_64-highs"