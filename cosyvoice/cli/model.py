# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import logging
from typing import Generator
import queue
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm
from cosyvoice.utils.common import TrtContextWrapper


class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}
        self.silent_tokens = []
        self.token_ready_event = {}  # Event for signaling tokens ready

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device, weights_only=True), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device, weights_only=True), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device, weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        cur_silent_token_num, max_silent_token_num = 0, 5
        # Use autocast only when fp16 is enabled and NOT using vllm (vllm handles its own precision)
        use_autocast = self.fp16 is True and not hasattr(self.llm, 'vllm')
        with self.llm_context, torch.cuda.amp.autocast(use_autocast):
            if isinstance(text, Generator):
                # bistream mode now supports vllm! vllm uses prefix cache for acceleration
                if self.__class__.__name__ == 'CosyVoiceModel':
                    raise ValueError('streaming input text is only implemented for CosyVoice2/3!')
                token_generator = self.llm.inference_bistream(text=text,
                                                              prompt_text=prompt_text.to(self.device),
                                                              prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                              prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                              prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                              embedding=llm_embedding.to(self.device))
            else:
                token_generator = self.llm.inference(text=text.to(self.device),
                                                     text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_text=prompt_text.to(self.device),
                                                     prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                     prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                     embedding=llm_embedding.to(self.device),
                                                     uuid=uuid)  
            for i in token_generator:
                if i in self.silent_tokens:
                    cur_silent_token_num += 1
                    if cur_silent_token_num > max_silent_token_num:
                        continue
                else:
                    cur_silent_token_num = 0
                self.tts_speech_token_dict[uuid].append(i)
                # Signal that new tokens are available
                if uuid in self.token_ready_event:
                    self.token_ready_event[uuid].set()
        self.llm_end_dict[uuid] = True
        # Final signal to wake up any waiting threads
        if uuid in self.token_ready_event:
            self.token_ready_event[uuid].set()

    def vc_job(self, source_speech_token, uuid):
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten().tolist()
        self.llm_end_dict[uuid] = True
        # Signal that tokens are ready
        if uuid in self.token_ready_event:
            self.token_ready_event[uuid].set()

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, self.flow_cache_dict[uuid] = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_token=prompt_token.to(self.device),
                                                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_feat=prompt_feat.to(self.device),
                                                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                                      embedding=embedding.to(self.device),
                                                                      flow_cache=self.flow_cache_dict[uuid])

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
            self.token_ready_event[this_uuid] = threading.Event()
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            event = self.token_ready_event[this_uuid]
            while True:
                # Wait for event signal (no polling!)
                event.wait()
                event.clear()
                
                # Process all available chunks
                while len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
            self.token_ready_event.pop(this_uuid, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()


class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 25
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.flow_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        self.silent_tokens = []
        # Pipeline parallel related - event-driven
        self.flow_input_queue = {}
        self.flow_output_queue = {}
        self.flow_end_dict = {}
        self.token_ready_event = {}  # Event to signal tokens are ready

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_vllm(self, model_dir):
        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        from vllm import EngineArgs, LLMEngine
        engine_args = EngineArgs(model=model_dir,
                                 skip_tokenizer_init=True,
                                 enable_prompt_embeds=True,
                                 gpu_memory_utilization=0.2)
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers

    @torch.inference_mode()
    def warmup(self):
        """Warmup vLLM, flow and hift to reduce first inference latency."""
        logging.info('Warming up model components...')
        
        # Warmup flow and hift
        logging.info('Warming up flow and hift...')
        dummy_token = torch.zeros(1, 50, dtype=torch.int32).to(self.device)
        dummy_prompt_token = torch.zeros(1, 10, dtype=torch.int32).to(self.device)
        dummy_prompt_feat = torch.zeros(1, 100, self.flow.output_size).to(self.device)
        dummy_embedding = torch.randn(1, 192).to(self.device)
        
        with torch.cuda.amp.autocast(self.fp16):
            dummy_mel, _ = self.flow.inference(
                token=dummy_token,
                token_len=torch.tensor([50], dtype=torch.int32).to(self.device),
                prompt_token=dummy_prompt_token,
                prompt_token_len=torch.tensor([10], dtype=torch.int32).to(self.device),
                prompt_feat=dummy_prompt_feat,
                prompt_feat_len=torch.tensor([100], dtype=torch.int32).to(self.device),
                embedding=dummy_embedding,
                streaming=False,
                finalize=True
            )
            # Warmup hift
            dummy_cache_source = torch.zeros(1, 1, 0).to(self.device)
            self.hift.inference(speech_feat=dummy_mel, cache_source=dummy_cache_source)
        
        # Warmup vLLM if loaded
        if hasattr(self.llm, 'vllm'):
            logging.info('Warming up vLLM...')
            from vllm import SamplingParams
            dummy_embeds = torch.randn(10, self.llm.llm_input_size, device=self.device, dtype=torch.bfloat16)
            sampling_params = SamplingParams(max_tokens=1, top_k=1)
            with self.llm.lock:
                self.llm.vllm.add_request("warmup", {"prompt_embeds": dummy_embeds}, sampling_params)
                # Run one step to warm up
                self.llm.vllm.step()
                # Abort the request
                self.llm.vllm.abort_request("warmup")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        logging.info('Warmup completed.')

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        # Flow inference timing
        flow_start = time.time()
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        flow_time = time.time() - flow_start
        
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        # HiFT inference timing
        hift_start = time.time()
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        hift_time = time.time() - hift_start
        
        logging.info(f'token2wav: flow={flow_time*1000:.1f}ms, hift={hift_time*1000:.1f}ms, tokens={token.shape[1]}')
        return tts_speech

    def flow_job(self, flow_prompt_speech_token, prompt_speech_feat, flow_embedding, this_uuid, stream):
        """Flow thread job: processes tokens from queue and produces audio chunks.
        
        This method runs in a separate thread, consuming tokens from flow_input_queue
        (blocking wait, no polling) and producing audio to flow_output_queue.
        """
        token_offset = 0
        prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
        
        with self.flow_context:
            while True:
                # Blocking wait for work item (no polling!)
                work_item = self.flow_input_queue[this_uuid].get()
                
                if work_item is None:  # Sentinel to stop
                    break
                
                token_list, is_final = work_item
                this_tts_speech_token = torch.tensor(token_list).unsqueeze(dim=0)
                
                this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
                
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                 prompt_token=flow_prompt_speech_token,
                                                 prompt_feat=prompt_speech_feat,
                                                 embedding=flow_embedding,
                                                 token_offset=token_offset,
                                                 uuid=this_uuid,
                                                 stream=stream,
                                                 finalize=is_final)
                
                if not is_final:
                    token_offset += this_token_hop_len
                
                # Put result into output queue
                self.flow_output_queue[this_uuid].put({'tts_speech': this_tts_speech.cpu()})
        
        self.flow_end_dict[this_uuid] = True
        # Signal end with sentinel
        self.flow_output_queue[this_uuid].put(None)

    def dispatcher_job(self, flow_prompt_speech_token, this_uuid, start_time):
        """Dispatcher thread job: collects tokens from LLM and dispatches to Flow.
        
        Uses event-driven approach (no polling). Waits for token_ready_event signal
        from LLM thread, then batches tokens for Flow processing.
        """
        token_offset = 0
        prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
        event = self.token_ready_event[this_uuid]
        first_chunk_logged = False
        
        while True:
            this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
            required_len = this_token_hop_len + self.flow.pre_lookahead_len
            
            # Wait for event signal (no polling!)
            event.wait()
            event.clear()
            
            current_tokens = len(self.tts_speech_token_dict.get(this_uuid, []))
            llm_done = self.llm_end_dict.get(this_uuid, False)
            
            # Dispatch all available chunks
            while current_tokens - token_offset >= required_len:
                if not first_chunk_logged:
                    llm_time = (time.time() - start_time) * 1000
                    logging.info(f'llm: {required_len}_tokens={llm_time:.1f}ms')
                    first_chunk_logged = True
                
                token_list = self.tts_speech_token_dict[this_uuid][:token_offset + required_len]
                self.flow_input_queue[this_uuid].put((token_list, False))
                token_offset += this_token_hop_len
                this_token_hop_len = self.token_hop_len
                required_len = this_token_hop_len + self.flow.pre_lookahead_len
                current_tokens = len(self.tts_speech_token_dict.get(this_uuid, []))
            
            # Exit when LLM is done
            if llm_done and current_tokens - token_offset < required_len:
                break
        
        # Process final remaining tokens
        remaining_tokens = self.tts_speech_token_dict.get(this_uuid, [])
        if len(remaining_tokens) > 0:
            self.flow_input_queue[this_uuid].put((remaining_tokens, True))
        
        # Signal end to flow thread
        self.flow_input_queue[this_uuid].put(None)

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0,
            pipeline_parallel=False, **kwargs):
        """TTS inference with optional pipeline parallelism.
        
        Args:
            pipeline_parallel: If True, enables three-thread pipeline parallelism where
                              LLM, Dispatcher, and Flow run in separate threads for
                              maximum throughput. This is especially useful with vLLM bistream.
        """
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            # Event for signaling new tokens are ready (event-driven, no polling)
            self.token_ready_event[this_uuid] = threading.Event()
        
        # Enable pipeline parallel for bistream mode with vllm by default
        is_bistream = isinstance(text, Generator)
        has_vllm = hasattr(self.llm, 'vllm')
        if is_bistream and has_vllm and stream:
            pipeline_parallel = True
            logging.info('vLLM bistream mode enabled with pipeline parallelism')
        
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        
        if stream is True:
            if pipeline_parallel:
                # Pipeline parallel mode: LLM, Dispatcher, Flow run in parallel threads
                yield from self._tts_pipeline_parallel(
                    this_uuid, p, flow_prompt_speech_token, prompt_speech_feat, flow_embedding, stream
                )
            else:
                # Event-driven sequential mode (no polling)
                yield from self._tts_sequential(
                    this_uuid, p, flow_prompt_speech_token, prompt_speech_feat, flow_embedding, stream
                )
        else:
            # Non-stream mode: wait for all tokens, then process
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=0,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.token_ready_event.pop(this_uuid, None)
            # Cleanup pipeline parallel queues if used
            self.flow_input_queue.pop(this_uuid, None)
            self.flow_output_queue.pop(this_uuid, None)
            self.flow_end_dict.pop(this_uuid, None)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()

    def _tts_sequential(self, this_uuid, llm_thread, flow_prompt_speech_token, prompt_speech_feat, flow_embedding, stream):
        """Event-driven sequential TTS mode: LLM generates, then Flow processes.
        
        Uses threading.Event for synchronization instead of polling.
        """
        token_offset = 0
        prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
        event = self.token_ready_event[this_uuid]
        chunk_count = 0
        tts_start_time = time.time()
        
        while True:
            this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
            required_tokens = this_token_hop_len + self.flow.pre_lookahead_len
            
            # Wait for event signal (no polling!)
            wait_start = time.time()
            event.wait()
            event.clear()
            wait_time = (time.time() - wait_start) * 1000
            
            current_tokens = len(self.tts_speech_token_dict[this_uuid])
            llm_done = self.llm_end_dict[this_uuid]
            
            # Process all available chunks
            while current_tokens - token_offset >= required_tokens:
                chunk_count += 1
                if chunk_count == 1:
                    llm_time = (time.time() - tts_start_time) * 1000
                    logging.info(f'llm: {required_tokens}_tokens={llm_time:.1f}ms (wait={wait_time:.1f}ms)')
                
                this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + required_tokens]).unsqueeze(dim=0)
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                 prompt_token=flow_prompt_speech_token,
                                                 prompt_feat=prompt_speech_feat,
                                                 embedding=flow_embedding,
                                                 token_offset=token_offset,
                                                 uuid=this_uuid,
                                                 stream=stream,
                                                 finalize=False)
                token_offset += this_token_hop_len
                yield {'tts_speech': this_tts_speech.cpu()}
                # Update for next iteration
                this_token_hop_len = self.token_hop_len  # No padding after first chunk
                required_tokens = this_token_hop_len + self.flow.pre_lookahead_len
                current_tokens = len(self.tts_speech_token_dict[this_uuid])
            
            # Exit when LLM is done and not enough tokens for another chunk
            if llm_done and current_tokens - token_offset < required_tokens:
                break
        
        llm_thread.join()
        # Deal with remaining tokens
        this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                         prompt_token=flow_prompt_speech_token,
                                         prompt_feat=prompt_speech_feat,
                                         embedding=flow_embedding,
                                         token_offset=token_offset,
                                         uuid=this_uuid,
                                         finalize=True)
        yield {'tts_speech': this_tts_speech.cpu()}

    def _tts_pipeline_parallel(self, this_uuid, llm_thread, flow_prompt_speech_token, prompt_speech_feat, flow_embedding, stream):
        """Pipeline parallel TTS mode: LLM, Dispatcher, Flow run in parallel threads.
        
        Event-driven design (no polling):
        - LLM thread: generates speech tokens, signals token_ready_event
        - Dispatcher thread: waits for event, batches tokens, puts to flow_input_queue
        - Flow thread: blocking wait on flow_input_queue, processes, puts to flow_output_queue
        - Main thread: blocking wait on flow_output_queue
        
        All three run simultaneously, maximizing throughput.
        """
        start_time = time.time()
        
        # Initialize queues for this session
        with self.lock:
            self.flow_input_queue[this_uuid] = queue.Queue()
            self.flow_output_queue[this_uuid] = queue.Queue()
            self.flow_end_dict[this_uuid] = False
        
        # Start dispatcher thread
        dispatcher_thread = threading.Thread(
            target=self.dispatcher_job,
            args=(flow_prompt_speech_token, this_uuid, start_time)
        )
        dispatcher_thread.start()
        
        # Start flow thread
        flow_thread = threading.Thread(
            target=self.flow_job,
            args=(flow_prompt_speech_token, prompt_speech_feat, flow_embedding, this_uuid, stream)
        )
        flow_thread.start()
        
        # Yield audio chunks as they become available (blocking wait, no polling!)
        while True:
            result = self.flow_output_queue[this_uuid].get()  # Blocking wait
            if result is None:  # Sentinel indicating end
                break
            yield result
        
        # Wait for all threads to complete
        llm_thread.join()
        dispatcher_thread.join()
        flow_thread.join()


class CosyVoice3Model(CosyVoice2Model):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 25
        # rtf and decoding related
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.flow_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        # FSQ silent and breath token
        self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
        # Pipeline parallel related - event-driven
        self.flow_input_queue = {}
        self.flow_output_queue = {}
        self.flow_end_dict = {}
        self.token_ready_event = {}

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        # Flow inference timing
        flow_start = time.time()
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        flow_time = time.time() - flow_start
        
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append mel cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel = self.hift_cache_dict[uuid]['mel']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
            self.hift_cache_dict[uuid]['mel'] = tts_mel
        else:
            self.hift_cache_dict[uuid] = {'mel': tts_mel, 'speech_offset': 0}
        if speed != 1.0:
            assert token_offset == 0 and finalize is True, 'speed change only support non-stream inference mode'
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
        
        # HiFT inference timing
        hift_start = time.time()
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        hift_time = time.time() - hift_start
        
        tts_speech = tts_speech[:, self.hift_cache_dict[uuid]['speech_offset']:]
        self.hift_cache_dict[uuid]['speech_offset'] += tts_speech.shape[1]
        
        logging.info(f'token2wav: flow={flow_time*1000:.1f}ms, hift={hift_time*1000:.1f}ms, tokens={token.shape[1]}')
        return tts_speech

    @torch.inference_mode()
    def warmup(self):
        """Warmup vLLM, flow and hift (CausalHiFTGenerator) to reduce first inference latency."""
        logging.info('Warming up model components...')
        
        # Warmup flow and hift
        logging.info('Warming up flow and hift...')
        dummy_token = torch.zeros(1, 50, dtype=torch.int32).to(self.device)
        dummy_prompt_token = torch.zeros(1, 10, dtype=torch.int32).to(self.device)
        dummy_prompt_feat = torch.zeros(1, 100, self.flow.output_size).to(self.device)
        dummy_embedding = torch.randn(1, 192).to(self.device)
        
        with torch.cuda.amp.autocast(self.fp16):
            dummy_mel, _ = self.flow.inference(
                token=dummy_token,
                token_len=torch.tensor([50], dtype=torch.int32).to(self.device),
                prompt_token=dummy_prompt_token,
                prompt_token_len=torch.tensor([10], dtype=torch.int32).to(self.device),
                prompt_feat=dummy_prompt_feat,
                prompt_feat_len=torch.tensor([100], dtype=torch.int32).to(self.device),
                embedding=dummy_embedding,
                streaming=False,
                finalize=True
            )
            # Warmup hift (CausalHiFTGenerator with finalize parameter)
            self.hift.inference(speech_feat=dummy_mel, finalize=True)
        
        # Warmup vLLM if loaded
        if hasattr(self.llm, 'vllm'):
            logging.info('Warming up vLLM...')
            from vllm import SamplingParams
            dummy_embeds = torch.randn(10, self.llm.llm_input_size, device=self.device, dtype=torch.bfloat16)
            sampling_params = SamplingParams(max_tokens=1, top_k=1)
            with self.llm.lock:
                self.llm.vllm.add_request("warmup", {"prompt_embeds": dummy_embeds}, sampling_params)
                # Run one step to warm up
                self.llm.vllm.step()
                # Abort the request
                self.llm.vllm.abort_request("warmup")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        logging.info('Warmup completed.')
