import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from pydub import AudioSegment
from .utils import filename, str2bool, write_srt
import io
import concurrent.futures
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="要轉錄的影片文件路徑")
    parser.add_argument("--model", default="large-v3",
                        choices=whisper.available_models(), help="要使用的 Whisper 模型名稱")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="儲存輸出檔案的目錄")
    parser.add_argument("--output_srt", type=str2bool, default=False,
                        help="是否在影片檔案旁邊產生 .srt 字幕檔案")
    parser.add_argument("--srt_only", type=str2bool, default=True,
                        help="只產生 .srt 字幕檔案，不產生疊加影片")
    parser.add_argument("--verbose", type=str2bool, default=True,
                        help="是否顯示進度條和調試信息")

    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="是否進行 X->X 語音識別 ('transcribe') 或 X->英文翻譯 ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
    help="影片的原始語言。如果未設置，則自動檢測。")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    video_paths: list = args.pop("video")
    
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} 是一個僅支援英文的模型，強制使用英文偵測。")
        args["language"] = "en"
    # 如果使用翻譯任務且設定了語言參數，則使用該語言
    elif language != "auto":
        args["language"] = language
        
    print(f"正在加載 Whisper 模型 {model_name}...")
    model = whisper.load_model(model_name)
    print("模型加載完成。")
    
    for video_path in video_paths:
        process_video(video_path, output_srt, srt_only, output_dir, model, args)

    print(f"\n所有檔案處理完成！共處理了 {len(video_paths)} 個檔案。")

def get_audio(video_path):
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"{filename(video_path)}.wav")

    ffmpeg.input(video_path).output(
        output_path,
        acodec="pcm_s16le", ac=1, ar="16k"
    ).run(quiet=True, overwrite_output=True)

    return output_path


def split_audio(audio_path, min_silence_len=1000, silence_thresh=-40, keep_silence=300):
    audio = AudioSegment.from_wav(audio_path)
    
    # 使用 numpy 進行音訊處理
    samples = np.array(audio.get_array_of_samples())
    
    # 計算音量
    chunk_size = int(audio.frame_rate * (min_silence_len / 1000.0))
    volume = np.array([max(chunk) for chunk in np.array_split(np.abs(samples), len(samples) // chunk_size)])
    
    # 找到靜音部分
    silent = volume < (10 ** (silence_thresh / 20.0) * audio.max_possible_amplitude)
    
    # 分割音訊
    splits = np.where(np.diff(silent.astype(int)))[0]
    segments = np.split(samples, splits)
    
    temp_segments = []
    start_times = []
    current_time = 0

    for i, segment in enumerate(segments):
        if not silent[i]:
            temp_segment = AudioSegment(segment.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width, channels=audio.channels)
            temp_segment_path = f"{audio_path}_segment_{i}.wav"
            temp_segment.export(temp_segment_path, format="wav")
            temp_segments.append(temp_segment_path)
            start_times.append(current_time / 1000.0)
        current_time += len(segment) / audio.frame_rate * 1000

    return temp_segments, start_times

def get_subtitles(audio_path, output_srt, output_dir, model, args, video_path):
    srt_path = output_dir if output_srt else tempfile.gettempdir()
    srt_path = os.path.join(srt_path, f"{filename(video_path)}.srt")

    # 使用靜音檢測分割音訊
    audio_segments, start_times = split_audio(audio_path)
    
    # 初始化用於儲存分段結果的列表
    all_segments = []

     # 並行處理每個音訊片段
    print(f"開始並行處理 {len(audio_segments)} 個音訊片段...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_audio_segment, segment, start_time, model, args) 
                   for segment, start_time in zip(audio_segments, start_times)]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            all_segments.extend(future.result())
            completed += 1
            print(f"已完成 {completed}/{len(audio_segments)} 個音訊片段的處理")

    print("所有音訊片段處理完成，正在生成 SRT 文件...")

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
        print(f"\n處理檔案: {filename(video_path)}")
        
        # 擷取音訊
        print(f"正在從 {filename(video_path)} 提取音訊...")
        audio_path = get_audio(video_path)
        
        # 產生字幕
        print(f"正在為 {filename(video_path)} 生成字幕...")
        srt_path = get_subtitles(audio_path, output_srt or srt_only, output_dir, model, args, video_path)

        if srt_only:
            print(f"已生成 {filename(video_path)} 的字幕文件：{srt_path}")
            return

        out_path = os.path.join(output_dir, f"{filename(video_path)}_subtitled.mp4")

        print(f"正在為 {filename(video_path)} 添加字幕...")

        video = ffmpeg.input(video_path)
        audio = video.audio

        ffmpeg.concat(
            video.filter('subtitles', srt_path, force_style="OutlineColour=&H40000000,BorderStyle=3"), audio, v=1, a=1
        ).output(out_path).run(quiet=True, overwrite_output=True)

        print(f"已將字幕添加到 {os.path.abspath(out_path)}。")

        print(f"完成處理 {filename(video_path)}。")

    except Exception as e:
        print(f"處理 {filename(video_path)} 時發生錯誤: {str(e)}")
   

def process_audio_segment(segment, start_time, model, args):
    result = model.transcribe(segment, **args)
    for seg in result["segments"]:
        seg["start"] += start_time
        seg["end"] += start_time
    return result["segments"]

if __name__ == '__main__':
    main()