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

以下指令將產生一個 `subtitled/video.mp4` 的檔案，其中包含帶有新增字幕的輸入影片。

    auto_subtitle /path/to/video.mp4 -o subtitled/

預設設定（選擇`small`模型）適合用在轉錄英文。您可以選擇使用更大的模型以取得更好的结果（尤其是使用其他语言）。可用的模型有 `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`, `large-v1`, `large-v2`, `large-v3`.

    auto_subtitle /path/to/video.mp4 --model medium

新增 `--task translate` 會將字幕翻譯成英文:

    auto_subtitle /path/to/video.mp4 --task translate

執行以下命令可以查看所有可用選項:

    auto_subtitle --help

## License

This script is open-source and licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.
