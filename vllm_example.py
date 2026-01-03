import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm
import torchaudio

def cosyvoice2_example():
    """ CosyVoice2 vllm usage
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    for i in tqdm(range(100)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', stream=False)):
            continue


def cosyvoice3_bistream_vllm_example():
    """ CosyVoice3 bistream usage with vLLM acceleration
        This example demonstrates the new vLLM bistream feature with:
        1. vLLM KV cache: Independent requests are submitted and incremental results are collected
        2. Pipeline parallelism: LLM, Dispatcher, and Flow run in separate threads
        
        When vLLM is enabled with bistream mode, prefix caching is used for acceleration.
        Pipeline parallel is automatically enabled - vLLM and Flow run in parallel for better performance.
    """
    import torch
    import time
    
    # Load model with vLLM enabled
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)

    def text_generator():
        """Simulate streaming text input (e.g., from a text LLM)"""
        yield '收到好'
        yield '友从远方'
        yield '寄来的生日礼物，'
        yield '那份意外'
        yield '的惊喜与深'
        yield '深的祝福'
        yield '让我心中充满'
        yield '了甜蜜的快乐，'
        yield '笑容如花儿般'
        yield '绽放。'

    # When vLLM is enabled with Generator input:
    # - vLLM bistream inference is used with prefix caching
    # - Pipeline parallel is automatically enabled
    # - LLM generates tokens while Flow renders audio simultaneously
    speech_chunks = []
    chunk_times = []
    start_time = time.time()
    first_chunk_time = None
    chunk_count = 0
    
    for wav in cosyvoice.inference_zero_shot(text_generator(), 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', stream=False):
        chunk_time = time.time()
        if first_chunk_time is None:
            first_chunk_time = chunk_time
            print('\n>>> First audio chunk latency: {:.3f}s'.format(first_chunk_time - start_time))
        chunk_count += 1
        chunk_times.append(chunk_time - start_time)
        speech_chunks.append(wav['tts_speech'])
        print(f'    Chunk {chunk_count}: +{chunk_time - start_time:.3f}s, samples={wav["tts_speech"].shape[1]}')
    
    end_time = time.time()
    tts_speech = torch.cat(speech_chunks, dim=1)
    torchaudio.save('sft_bistream_vllm.wav', tts_speech, cosyvoice.sample_rate)
    
    # Print statistics
    audio_duration = tts_speech.shape[1] / cosyvoice.sample_rate
    print('\n' + '=' * 60)
    print('Statistics:')
    print('=' * 60)
    print('>>> Total chunks: {}'.format(chunk_count))
    print('>>> Audio duration: {:.3f}s'.format(audio_duration))
    print('>>> Total inference time: {:.3f}s'.format(end_time - start_time))
    print('>>> RTF (real-time factor): {:.3f}'.format((end_time - start_time) / audio_duration))
    print('>>> Saved to: sft_bistream_vllm.wav')


def main():
    # cosyvoice2_example()
    cosyvoice3_bistream_vllm_example()


if __name__ == '__main__':
    main()
