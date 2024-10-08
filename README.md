# 在你的影片中自動新增字幕

這個儲存庫使用 `ffmpeg` 以及 [OpenAI's Whisper](https://openai.com/blog/whisper) 在任何影片上自動產生和新增字幕。

## 安裝

首先，您需要 Python 3.7 或更高版本。透過執行以下指令安装：

    pip install git+https://github.com/SZZhang0619/auto-subtitle.git

您還需要安裝 [`ffmpeg`](https://ffmpeg.org/), 它可從大部分套件管理器中取得:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg
```

## 用法

### 基本用法

以下指令將處理一個或多個影片檔案，並在指定的輸出目錄中生成帶有嵌入字幕的影片：

    auto_subtitle /path/to/video1.mp4 /path/to/video2.mp4 -o output_directory/

### 選擇模型

您可以選擇不同的模型來優化轉錄效果。可用的模型有 `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3`：

    auto_subtitle /path/to/video.mp4 --model large-v3

### 翻譯任務

使用 `--task translate` 將字幕翻譯成英文：

    auto_subtitle /path/to/video.mp4 --task translate

### 只生成 SRT 檔案

如果您只需要 SRT 字幕檔案而不需要嵌入字幕的影片：

    auto_subtitle /path/to/video.mp4 --srt_only True

### 生成單獨的 SRT 檔案

在生成帶字幕的影片的同時，也生成單獨的 SRT 檔案：

    auto_subtitle /path/to/video.mp4 --output_srt True

### 選擇計算類型

根據您的硬體和性能需求，選擇適合的計算類型（float32, float16, 或 int8）：

    auto_subtitle /path/to/video.mp4 --compute_type float16

### 指定語言

如果您知道影片的原始語言，可以指定語言來提高準確性：

    auto_subtitle /path/to/video.mp4 --language zh

### 選擇設備

指定使用 CPU 或 CUDA 進行處理：

    auto_subtitle /path/to/video.mp4 --device cuda

### 調整詳細程度

增加輸出的詳細程度以獲得更多處理信息（可以使用多次，如 -vv）：

    auto_subtitle /path/to/video.mp4 -v

### 查看所有選項

執行以下命令可以查看所有可用選項：

    auto_subtitle --help

注意：如果不指定輸入檔案，程序將自動處理當前目錄下所有支援的檔案（.mp4, .mp3, .m4a）。

## License

This script is open-source and licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.