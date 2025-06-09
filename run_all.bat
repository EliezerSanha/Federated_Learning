@echo off
echo Iniciando servidor e clientes...

:: Caminho base onde estão os arquivos .py
set BASE_DIR=C:\Users\Pichau\Desktop\Estudos PPGCC\06 Cursos Dev\Federated Learning

:: Ativa o ambiente Conda
call conda activate federated_pytorch

:: Abre uma janela para o servidor
start cmd /k python "%BASE_DIR%\server.py"

:: Aguarda 3 segundos para o servidor iniciar
timeout /t 3 >nul

:: Abre uma janela para cada cliente
start cmd /k python "%BASE_DIR%\client_1.py"
start cmd /k python "%BASE_DIR%\client_2.py"
start cmd /k python "%BASE_DIR%\client_3.py"
