import os
import ffmpeg
from faster_whisper import WhisperModel
import argparse
import warnings
import tempfile
from pydub import AudioSegment
from .utils import filename, str2bool, write_srt
import io
import concurrent.futures
import numpy as np
import logging
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="要轉錄的影片文件路徑")
    parser.add_argument("--model", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"], 
                        help="要使用的 Whisper 模型名稱")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="儲存輸出檔案的目錄")
    parser.add_argument("--output_srt", type=str2bool, default=False,
                        help="是否在影片檔案旁邊產生 .srt 字幕檔案")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="只產生 .srt 字幕檔案，不產生疊加影片")
    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="是否進行 X->X 語音識別 ('transcribe') 或 X->英文翻譯 ('translate')")
    parser.add_argument("--language", type=str, default="auto", 
                        help="影片的原始語言。如果未設置，則自動檢測。")
    parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                        help="使用的設備 (cuda 或 cpu)")
    parser.add_argument("--verbose", "-v", action="count", default=0,
                        help="增加輸出的詳細程度 (可以使用多次，例如 -vv)")
    parser.add_argument("--compute_type", type=str, default="float32",
                        choices=["float32", "float16", "int8"],
                        help="計算類型 (float32, float16, 或 int8)")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    video_paths: list = args.pop("video")
    device: str = args.pop("device")
    verbose: int = args.pop("verbose")
    compute_type: str = args.pop("compute_type")
    
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} 是一個僅支援英文的模型，強制使用英文偵測。")
        args["language"] = "en"
    # 如果使用翻譯任務且設定了語言參數，則使用該語言
    elif language != "auto":
        args["language"] = language
        
    # 設置日誌級別
    if verbose == 0:
        log_level = logging.WARNING
    elif verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print(f"正在加載 Whisper 模型 {model_name}...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print(f"模型加載完成。使用計算類型: {compute_type}")
    
    with tqdm(total=len(video_paths), desc="影片處理") as pbar:
        for video_path in video_paths:
            try:
                process_video(video_path, output_srt, srt_only, output_dir, model, args)
            except Exception as e:
                logging.error(f"處理 {video_path} 時發生錯誤: {str(e)}")
                print(f"處理 {video_path} 時發生錯誤。請查看日誌以獲取更多信息。")
            finally:
                pbar.update(1)

    print(f"\n所有檔案處理完成！共處理了 {len(video_paths)} 個檔案。")

def get_audio(video_path):
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"{filename(video_path)}.wav")

    ffmpeg.input(video_path).output(
        output_path,
        acodec="pcm_s16le", ac=1, ar="16k"
    ).run(quiet=True, overwrite_output=True)

    return output_path


def split_audio(audio_path, segment_length=300):  # 300秒 = 5分鐘
    audio = AudioSegment.from_wav(audio_path)
    
    # 計算總段數
    total_segments = int(np.ceil(len(audio) / (segment_length * 1000)))
    
    temp_segments = []
    start_times = []

    for i in range(total_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, len(audio) / 1000)
        
        segment = audio[start_time*1000:end_time*1000]
        temp_segment_path = f"{audio_path}_segment_{i}.wav"
        segment.export(temp_segment_path, format="wav")
        
        temp_segments.append(temp_segment_path)
        start_times.append(start_time)

    return temp_segments, start_times

def get_subtitles(audio_path, output_srt, output_dir, model, args, video_path):
    srt_path = output_dir if output_srt else tempfile.gettempdir()
    srt_path = os.path.join(srt_path, f"{filename(video_path)}.srt")

    # 使用固定時間間隔分割音訊
    audio_segments, start_times = split_audio(audio_path)
    
    # 初始化用於儲存分段結果的列表
    all_segments = []

    # 並行處理每個音訊片段
    logging.info(f"開始並行處理 {len(audio_segments)} 個音訊片段...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_audio_segment, segment, start_time, model, args) 
                   for segment, start_time in zip(audio_segments, start_times)]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            all_segments.extend(future.result())
            completed += 1
            logging.info(f"已完成 {completed}/{len(audio_segments)} 個音訊片段的處理")

    logging.info("所有音訊片段處理完成，正在生成 SRT 文件...")

    # 依照開始時間排序片段
    all_segments.sort(key=lambda x: x['start'])

    # 將所有片段寫入單一 SRT 檔案
    with io.StringIO() as buffer:
        write_srt(all_segments, file=buffer)
        srt_content = buffer.getvalue()

    with open(srt_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)

    # 清理臨時文件
    for segment in audio_segments:
        os.remove(segment)

    return srt_path

def process_video(video_path, output_srt, srt_only, output_dir, model, args):
    try:
        with tqdm(total=4, desc=f"處理 {filename(video_path)}", leave=False) as pbar:
            logging.info(f"\n處理檔案: {filename(video_path)}")
            
            # 擷取音訊
            logging.info(f"正在從 {filename(video_path)} 提取音訊...")
            audio_path = get_audio(video_path)
            pbar.update(1)
            
            # 產生字幕
            logging.info(f"正在為 {filename(video_path)} 生成字幕...")
            srt_path = get_subtitles(audio_path, output_srt or srt_only, output_dir, model, args, video_path)
            pbar.update(1)

            if srt_only:
                logging.info(f"已生成 {filename(video_path)} 的字幕文件：{srt_path}")
                pbar.update(2)
                return

            out_path = os.path.join(output_dir, f"{filename(video_path)}_subtitled.mp4")

            logging.info(f"正在為 {filename(video_path)} 添加字幕...")
            pbar.update(1)

            video = ffmpeg.input(video_path)
            audio = video.audio

            ffmpeg.concat(
                video.filter('subtitles', srt_path, force_style="OutlineColour=&H40000000,BorderStyle=3"), audio, v=1, a=1
            ).output(out_path).run(quiet=True, overwrite_output=True)

            logging.info(f"已將字幕添加到 {os.path.abspath(out_path)}。")
            pbar.update(1)

            logging.info(f"完成處理 {filename(video_path)}。")

    except FileNotFoundError as e:
        logging.error(f"處理 {filename(video_path)} 時發生錯誤: 找不到文件 - {str(e)}")
    except ffmpeg.Error as e:
        logging.error(f"處理 {filename(video_path)} 時發生 FFmpeg 錯誤: {str(e)}")
    except IOError as e:
        logging.error(f"處理 {filename(video_path)} 時發生 I/O 錯誤: {str(e)}")
    except ValueError as e:
        logging.error(f"處理 {filename(video_path)} 時發生值錯誤: {str(e)}")
    except Exception as e:
        logging.error(f"處理 {filename(video_path)} 時發生未預期的錯誤: {str(e)}")

def process_audio_segment(segment, start_time, model, args):
    segments, _ = model.transcribe(segment, **args)
    result = []
    for seg in segments:
        result.append({
            "start": seg.start + start_time,
            "end": seg.end + start_time,
            "text": seg.text
        })
    return result

if __name__ == '__main__':
    main()