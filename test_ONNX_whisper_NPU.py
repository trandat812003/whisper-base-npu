import numpy as np
import onnxruntime as ort
import librosa
from transformers import WhisperProcessor

def extract_features(wav_path: str, processor: WhisperProcessor):
    audio, sr = librosa.load(wav_path, sr=16000)
    feats = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="np")
    return feats["input_features"].astype(np.float32)

def run_encoder(encoder_sess: ort.InferenceSession, input_features: np.ndarray):
    name = encoder_sess.get_inputs()[0].name
    outs = encoder_sess.run(None, {name: input_features})
    return outs[0]

def run_decoder_once(decoder_sess: ort.InferenceSession, input_ids: np.ndarray, encoder_hidden: np.ndarray):
    inps = decoder_sess.get_inputs()
    if len(inps) < 2:
        raise RuntimeError("Decoder ONNX kỳ vọng ít nhất 2 inputs (input_ids, encoder_hidden). Kiểm tra model.")
    feed = {inps[0].name: input_ids, inps[1].name: encoder_hidden}
    outs = decoder_sess.run(None, feed)
    logits = outs[0] 
    return logits

def greedy_decode_with_forced(encoder_sess, decoder_sess, input_features: np.ndarray, processor: WhisperProcessor, forced_decoder_ids: np.ndarray, max_new_tokens=200):
    encoder_hidden = run_encoder(encoder_sess, input_features)

    if forced_decoder_ids.ndim == 1:
        input_ids = forced_decoder_ids[None].astype(np.int64)
    else:
        input_ids = forced_decoder_ids.astype(np.int64)

    generated = list(input_ids[0].tolist())

    # breakpoint()

    for _ in range(max_new_tokens):
        logits = run_decoder_once(decoder_sess, np.array([generated], dtype=np.int64), encoder_hidden)
        next_logits = logits[:, -1, :]
        next_id = int(np.argmax(next_logits, axis=-1)[0])
        generated.append(next_id)
        if next_id == processor.tokenizer.eos_token_id:
            break

    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return text

def main():
    encoder_path = "whisper_encoder.onnx"
    decoder_path = "whisper_decoder.onnx"

    encoder_sess = ort.InferenceSession(encoder_path, providers=["QNNExecutionProvider"])
    decoder_sess = ort.InferenceSession(decoder_path, providers=["QNNExecutionProvider"])

    processor = WhisperProcessor.from_pretrained("hkab/whisper-base-vietnamese-finetuned")

    forced_tok = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    forced_np = np.array(forced_tok, dtype=np.int64)
    if forced_np.ndim == 2 and forced_np.shape[0] > 1:
        forced_np = forced_np[0]

    wav = "demo.wav"
    feats = extract_features(wav, processor)

    transcript = greedy_decode_with_forced(
        encoder_sess, decoder_sess, feats, processor, forced_np, max_new_tokens=200
    )
    print("Prediction ONNX:", transcript)

if __name__ == "__main__":
    main()
