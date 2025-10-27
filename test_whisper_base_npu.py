import numpy as np
import samplerate
import torch
import whisper  # type: ignore
import time
from transformers import WhisperProcessor, WhisperTokenizer
import json
from scipy import special as scipy_special  # type: ignore
import audio2numpy as a2n
import argparse

from qai_hub_models.models._shared.hf_whisper.model import (
    CHUNK_LENGTH,
    SAMPLE_RATE,
    MASK_NEG,
    MEAN_DECODE_LEN,
)
from qai_appbuilder import (QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig)


class Encoder(QNNContext):
    def Inference(self, input_data):
        input_datas=[input_data]
        output_data = super().Inference(input_datas) 
        num_layers = len(output_data) // 2  # = 12

        kv_cache_cross = []
        for i in range(num_layers):
            k = output_data[2 * i]
            v = output_data[2 * i + 1]
            k = k.reshape(8, 1, 64, 1500)
            v = v.reshape(8, 1, 1500, 64)
            kv_cache_cross.append((k, v))
        return tuple(kv_cache_cross)
        
class Decoder(QNNContext):
    def Inference(self, *input_datas):
        # input_datas=[input_datas]
        output_data = super().Inference(input_datas)
        return output_data
    
QNNConfig.Config(r"C:\Users\asus\Documents\datnt\NPU_task\demo_qualcomm\qai_libs", Runtime.HTP, LogLevel.WARN, ProfilingLevel.BASIC)


decoder = Decoder("whisper_decoder", r"C:\Users\asus\Documents\datnt\NPU_task\demo_qualcomm\models\HfWhisperDecoder.bin")
encoder = Encoder("whisper_encoder", r"C:\Users\asus\Documents\datnt\NPU_task\demo_qualcomm\models\HfWhisperEncoder.bin")
processor = WhisperProcessor.from_pretrained("hkab/whisper-base-vietnamese-finetuned")
tokenizer = WhisperTokenizer.from_pretrained("hkab/whisper-base-vietnamese-finetuned")
CONFIG_PATH = r"C:\Users\asus\Documents\datnt\NPU_task\config.json"  
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_dict = json.load(f)

def Inference(audio_path):
    # Read and preprocess the audio.
    audio, audio_sample_rate = a2n.audio_from_file(audio_path)

    # Burst the HTP.
    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

    # Run the inference.
    out_chunked_tokens: list[list[int]] = [
        transcribe_single_chunk(x)
        for x in chunk_and_resample_audio(audio, audio_sample_rate)
    ]
    out_tokens: list[int] = []
    for chunk_tokens in out_chunked_tokens: out_tokens.extend(chunk_tokens)

    result = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()

    # Reset the HTP.
    PerfProfile.RelPerfProfileGlobal()
    
    # show the generated text
    print("Transcription:",result)


def transcribe_single_chunk(audio: np.ndarray):
    input_features = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.cpu().numpy()
    # encoder
    kv_cache_cross = encoder.Inference(input_features)

    sot = config_dict.get("decoder_start_token_id", None)
    num_decoder_blocks = config_dict.get("decoder_layers", None)
    attention_dim = config_dict.get("d_model", None)
    num_decoder_heads = config_dict.get("decoder_attention_heads", None)
    mask_neg = config_dict.get("mask_neg", MASK_NEG)
    eot = config_dict.get("eos_token_id", None)

    # decoder
    output_ids = torch.tensor([[sot]])  # Start of transcript
    output_logits = []
    output_length = output_ids.shape[1]

    position_ids = torch.tensor([0], dtype=torch.int32)
    attention_mask = torch.full(
        (1, 1, 1, MEAN_DECODE_LEN),
        mask_neg,
        dtype=torch.float32,
    )

    # init kv_cache_self
    k_cache_self = torch.zeros(
        (
            num_decoder_heads,
            1,
            attention_dim // num_decoder_heads,
            MEAN_DECODE_LEN - 1,
        ),
        dtype=torch.float32,
    )
    v_cache_self = torch.zeros(
        (
            num_decoder_heads,
            1,
            MEAN_DECODE_LEN - 1,
            attention_dim // num_decoder_heads,
        ),
        dtype=torch.float32,
    )
    kv_cache_self = tuple(
        (k_cache_self, v_cache_self) for _ in range(num_decoder_blocks)
    )

    for n in range(MEAN_DECODE_LEN - 1):
        # get current token
        input_ids = output_ids[:, n : n + 1].to(torch.int32)

        # update attention_mask
        attention_mask[:, :, :, MEAN_DECODE_LEN - n - 1] = 0.0

        # flattened kv caches input
        flattened_kv_cache_self = tuple(
            item for sublist in kv_cache_self for item in sublist
        )
        flattened_kv_cache_cross = tuple(
            item for sublist in kv_cache_cross for item in sublist
        )

        # decode and update kv_cache_self
        decoder_input = (
            (input_ids, attention_mask)
            + flattened_kv_cache_self
            + flattened_kv_cache_cross
            + (position_ids,)
        )
        decoder_output = decoder.Inference(*decoder_input)
        if isinstance(decoder_output, tuple) and len(decoder_output) == 2:
            logits, kv_cache_self = decoder_output
        else:
            logits = decoder_output[0]
            kv_cache_self = tuple(
                decoder_output[i : i + 2] for i in range(1, len(decoder_output), 2)
            )

        logits = logits.reshape(1, 51865, 1, 1)
        logits = torch.from_numpy(logits)

        output_logits.append(logits.detach().clone())

        output_id = torch.argmax(logits, 1).squeeze(0)
        if len(output_logits) == (MEAN_DECODE_LEN - 1) or output_id == eot:
            output_ids = torch.cat((output_ids, output_id), -1)
            break
        if n >= output_length - 1:
            output_ids = torch.cat((output_ids, output_id), -1)

        position_ids += 1

    return output_ids[0].tolist()


def chunk_and_resample_audio(
    audio: np.ndarray,
    audio_sample_rate: int,
    model_sample_rate=SAMPLE_RATE,
    model_chunk_seconds=CHUNK_LENGTH,
) -> list[np.ndarray]:
    if audio_sample_rate != model_sample_rate:
        audio = samplerate.resample(audio, model_sample_rate / audio_sample_rate)
        audio_sample_rate = model_sample_rate

    number_of_full_length_audio_chunks = (
        audio.shape[0] // audio_sample_rate // model_chunk_seconds
    )
    last_sample_in_full_length_audio_chunks = (
        audio_sample_rate * number_of_full_length_audio_chunks * model_chunk_seconds
    )

    if number_of_full_length_audio_chunks == 0:
        return [audio]

    return [
        *np.array_split(
            audio[:last_sample_in_full_length_audio_chunks],
            number_of_full_length_audio_chunks,
        ),
        audio[last_sample_in_full_length_audio_chunks:],
    ]

from pathlib import Path
def main():
    parser = argparse.ArgumentParser(description="Run Whisper on Qualcomm NPU")
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Đường dẫn đến file âm thanh (wav/mp3/flac...)"
    )
    args = parser.parse_args()
    audio_path = Path(args.audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {audio_path}")

    print("🎧 File âm thanh:", audio_path)
    print("🚀 Bắt đầu inference trên Qualcomm NPU...")

    Inference(str(audio_path))

    print("✅ Hoàn tất inference!")


def test_decode():
    data = np.load("decoder_input_test.npz")

    # Giải nén theo đúng thứ tự
    decoder_input = [torch.from_numpy(data[f'arr_{i}']) for i in range(len(data.files))]

    # Gọi decoder inference trực tiếp
    logits, k_cache, v_cache = decoder.Inference(*decoder_input)

if __name__ == "__main__":
    # test_decode()
    main()
