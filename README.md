# DOGAI - Sistema de Detecção de Objetos em Vídeo

## 📋 Descrição

DOGAI é um sistema de detecção de objetos em tempo real que utiliza YOLOv8 e captura de tela para detectar objetos específicos em vídeo. O projeto é desenvolvido em C++ e utiliza OpenCV, ONNX Runtime e Windows Graphics Capture.

## 🚀 Funcionalidades

- **Captura de tela em tempo real** usando Windows Graphics Capture
- **Detecção de objetos** com modelo YOLOv8 personalizado
- **Interface visual** com bounding boxes e informações de confiança
- **Sistema de logging** configurável (apenas erros por padrão)
- **Configuração flexível** via arquivo INI

## 🛠️ Tecnologias Utilizadas

- **C++17** - Linguagem principal
- **OpenCV 4.12.0** - Processamento de imagem e interface gráfica
- **ONNX Runtime 1.22.1** - Execução de modelos de IA
- **Windows Graphics Capture** - Captura de tela
- **DirectX 11** - Aceleração de hardware
- **CMake** - Sistema de build

---

## 🔧 Pré-requisitos

### Software Necessário
- **Visual Studio 2022** (Build Tools ou Community)
- **CMake 3.16+**
- **Windows 10/11** (para Windows Graphics Capture)

### Bibliotecas Externas
- **OpenCV 4.12.0** instalado em `D:/softwares/opencv/build`
- **ONNX Runtime 1.22.1** instalado em `D:/softwares/onnxruntime/onnxruntime-win-x64-1.22.1`

## 🚀 Instalação e Compilação

### 1. Clone o repositório
```bash
git clone <url-do-repositorio>
cd dogai
```

### 2. Configure as dependências
Certifique-se de que as bibliotecas estão nos caminhos corretos:
- OpenCV: `D:/softwares/opencv/build`
- ONNX Runtime: `D:/softwares/onnxruntime/onnxruntime-win-x64-1.22.1`

> NOTA: obviamente que você mesmo quem definirá isso. Aqui está genérico como está no meu.

### 3. Compile o projeto
```bash
.\build_modular.bat
```

### 4. Execute o programa
```bash
cd build\bin\Release
.\video_object_detection.exe
```

