# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua, Shengqiang Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import queue
import random
import time
import threading
from typing import Dict, Optional, Callable, List, Generator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.mask import make_pad_mask


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        text_encoder_input_size: int,
        llm_input_size: int,
        llm_output_size: int,
        text_token_size: int,
        speech_token_size: int,
        text_encoder: torch.nn.Module,
        llm: torch.nn.Module,
        sampling: Callable,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(
            text_token_size, text_encoder_input_size
        )
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(), llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = self.speech_token_size
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(
            text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(
        self,
        sos_emb,
        embedding,
        text_token,
        text_token_len,
        task_id_emb,
        speech_token,
        speech_token_len,
    ):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(
            speech_token, speech_token_len.cpu(), batch_first=True
        )
        lm_input = [
            torch.concat(
                [
                    sos_emb.squeeze(dim=0),
                    embedding[i],
                    text_token[i],
                    task_id_emb.squeeze(dim=0),
                    speech_token[i],
                ],
                dim=0,
            )
            for i in range(len(text_token))
        ]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)
        embedding = batch["embedding"].to(device)

        # 1. prepare llm_target
        lm_target = [
            torch.tensor(
                [IGNORE_ID] * (2 + text_token_len[i])
                + speech_token[i, : speech_token_len[i]].tolist()
                + [self.speech_token_size]
            )
            for i in range(text_token.size(0))
        ]
        lm_target = pad_sequence(
            lm_target, batch_first=True, padding_value=IGNORE_ID
        ).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. sos and task_id
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(
            sos_emb,
            embedding,
            text_token,
            text_token_len,
            task_id_emb,
            speech_token,
            speech_token_len,
        )

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(
            logits.view(-1, self.speech_token_size + 1),
            lm_target,
            ignore_label=IGNORE_ID,
        )
        return {"loss": loss, "acc": acc}

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        decoded_tokens: List,
        sampling: int,
        ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (top_ids < self.speech_token_size):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError(
                    "sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!".format(
                        max_trials
                    )
                )
        return top_ids

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = "",
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = (
                torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype)
                .to(device)
                .to(text.dtype)
            )

        # 3. concat llm_input
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=text.dtype
            ).to(device)
        lm_input = torch.concat(
            [sos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1
        )

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = (
            torch.zeros((0, 0, 0, 0), device=lm_input.device),
            torch.zeros((0, 0, 0, 0), device=lm_input.device),
        )
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(
                lm_input,
                offset=offset,
                required_cache_size=-1,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                att_mask=torch.tril(
                    torch.ones(
                        (1, lm_input.shape[1], lm_input.shape[1]),
                        device=lm_input.device,
                    )
                ).to(torch.bool),
            )
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0),
                out_tokens,
                sampling,
                ignore_eos=True if i < min_len else False,
            )
            if top_ids == self.eos_token:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


class Qwen2LM(TransformerLM):
    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        llm: torch.nn.Module,
        sampling: Callable,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        mix_ratio: List[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = speech_token_size
        self.fill_token = speech_token_size + 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(
            speech_token_size + 3, llm_input_size
        )

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(3)]
        self.vllm_output_queue = {}
        self.vllm_output_event = {}

    def prepare_lm_input_target(
        self,
        sos_emb,
        text_token,
        text_token_emb,
        text_token_len,
        task_id_emb,
        speech_token,
        speech_token_emb,
        speech_token_len,
    ):
        lm_target, lm_input = [], []
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(
            speech_token, speech_token_len.cpu(), batch_first=True
        )
        text_token_emb = unpad_sequence(
            text_token_emb, text_token_len.cpu(), batch_first=True
        )
        speech_token_emb = unpad_sequence(
            speech_token_emb, speech_token_len.cpu(), batch_first=True
        )
        for i in range(len(text_token)):
            # bistream sequence
            if (
                random.random() < 0.5
                and speech_token_len[i] / text_token_len[i]
                > self.mix_ratio[1] / self.mix_ratio[0]
            ):
                this_lm_target, this_lm_input = [], []
                this_lm_target.append(IGNORE_ID)
                this_lm_input.append(sos_emb.squeeze(dim=0))
                for j in range(
                    ((text_token_len[i] + 1) / self.mix_ratio[0]).ceil().int().item()
                ):
                    this_text_token = text_token[i][
                        j * self.mix_ratio[0] : (j + 1) * self.mix_ratio[0]
                    ].tolist()
                    this_speech_token = speech_token[i][
                        j * self.mix_ratio[1] : (j + 1) * self.mix_ratio[1]
                    ].tolist()
                    if len(this_text_token) == self.mix_ratio[0]:
                        assert len(this_speech_token) == self.mix_ratio[1]
                        this_lm_target += [IGNORE_ID] * (self.mix_ratio[0] - 1)
                        this_lm_target += this_speech_token
                        this_lm_target.append(self.fill_token)
                        this_lm_input.append(
                            text_token_emb[i][
                                j * self.mix_ratio[0] : (j + 1) * self.mix_ratio[0]
                            ]
                        )
                        this_lm_input.append(
                            speech_token_emb[i][
                                j * self.mix_ratio[1] : (j + 1) * self.mix_ratio[1]
                            ]
                        )
                    else:
                        this_lm_target += [-1] * len(this_text_token)
                        this_lm_target += speech_token[i][
                            j * self.mix_ratio[1] :
                        ].tolist()
                        this_lm_target.append(self.eos_token)
                        this_lm_input.append(text_token_emb[i][j * self.mix_ratio[0] :])
                        this_lm_input.append(task_id_emb.squeeze(dim=0))
                        this_lm_input.append(
                            speech_token_emb[i][j * self.mix_ratio[1] :]
                        )
                this_lm_target, this_lm_input = (
                    torch.tensor(this_lm_target),
                    torch.concat(this_lm_input, dim=0),
                )
            # unistream sequence
            else:
                this_lm_target = torch.tensor(
                    [IGNORE_ID] * (1 + text_token_len[i])
                    + speech_token[i].tolist()
                    + [self.eos_token]
                )
                this_lm_input = torch.concat(
                    [
                        sos_emb.squeeze(dim=0),
                        text_token_emb[i],
                        task_id_emb.squeeze(dim=0),
                        speech_token_emb[i],
                    ],
                    dim=0,
                )
            lm_target.append(this_lm_target)
            lm_input.append(this_lm_input)
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID)
        return lm_target, lm_input, lm_input_len

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)

        # 1. encode text_token
        text_token_emb = self.llm.model.model.embed_tokens(text_token)

        # 3. sos and task_id
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 2. encode speech_token
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(
            sos_emb,
            text_token,
            text_token_emb,
            text_token_len,
            task_id_emb,
            speech_token,
            speech_token_emb,
            speech_token_len,
        )
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target.to(device))
        acc = th_accuracy(
            logits.view(-1, self.speech_token_size + 3),
            lm_target,
            ignore_label=IGNORE_ID,
        )
        return {"loss": loss, "acc": acc}

    def forward_dpo(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)
        reject_speech_token = batch["reject_speech_token"].to(device)
        reject_speech_token_len = batch["reject_speech_token_len"].to(device)

        # 1. encode text_token
        text_token_emb = self.llm.model.model.embed_tokens(text_token)

        # 3. sos and task_id
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 2. encode speech_token
        speech_token = unpad_sequence(
            speech_token, speech_token_len.cpu(), batch_first=True
        )
        reject_speech_token = unpad_sequence(
            reject_speech_token, reject_speech_token_len.cpu(), batch_first=True
        )
        speech_token_combined = speech_token + reject_speech_token
        speech_token_combined = pad_sequence(
            speech_token_combined, batch_first=True, padding_value=0
        )
        speech_token_combined_len = torch.concat(
            [speech_token_len, reject_speech_token_len], dim=0
        )
        speech_token_combined_emb = self.speech_embedding(speech_token_combined)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(
            sos_emb,
            text_token.repeat(2, 1),
            text_token_emb.repeat(2, 1, 1),
            text_token_len.repeat(2),
            task_id_emb,
            speech_token_combined,
            speech_token_combined_emb,
            speech_token_combined_len,
        )
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        chosen_logits = logits[: text_token.shape[0]]
        rejected_logits = logits[text_token.shape[0] :]
        chosen_lm_target = lm_target[: text_token.shape[0]]
        rejected_lm_target = lm_target[text_token.shape[0] :]
        loss = self.criterion_ce(chosen_logits, chosen_lm_target.to(device))
        acc = th_accuracy(
            chosen_logits.view(-1, self.speech_token_size + 3),
            chosen_lm_target,
            ignore_label=IGNORE_ID,
        )

        # 5. calculate dpo logits
        chosen_lm_mask = chosen_lm_target == IGNORE_ID
        rejected_lm_mask = rejected_lm_target == IGNORE_ID
        chosen_logps = torch.gather(
            chosen_logits.log_softmax(dim=-1),
            dim=2,
            index=chosen_lm_target.masked_fill(chosen_lm_mask, 0).unsqueeze(dim=-1),
        ).squeeze(dim=-1)
        rejected_logps = torch.gather(
            rejected_logits.log_softmax(dim=-1),
            dim=2,
            index=rejected_lm_target.masked_fill(rejected_lm_mask, 0).unsqueeze(dim=-1),
        ).squeeze(dim=-1)
        chosen_logps = (chosen_logps * chosen_lm_mask).sum(dim=-1) / chosen_lm_mask.sum(
            dim=-1
        )
        rejected_logps = (rejected_logps * rejected_lm_mask).sum(
            dim=-1
        ) / rejected_lm_mask.sum(dim=-1)
        return {
            "loss": loss,
            "acc": acc,
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
        }

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = "",
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)

        # 3. concat llm_input
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=text.dtype
            ).to(device)
        lm_input = torch.concat(
            [sos_emb, text, task_id_emb, prompt_speech_token_emb], dim=1
        )

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token

    @torch.inference_mode()
    def inference_wrapper(self, lm_input, sampling, min_len, max_len, uuid):
        if hasattr(self, "vllm"):
            from vllm import SamplingParams, RequestOutput

            sampling_params = SamplingParams(
                top_k=sampling,
                stop_token_ids=self.stop_token_ids,
                min_tokens=min_len,
                max_tokens=max_len,
            )
            with self.lock:
                self.vllm.add_request(
                    uuid,
                    {
                        "prompt_embeds": lm_input.squeeze(0)
                        .to(torch.bfloat16)
                        .to(lm_input.device)
                    },
                    sampling_params,
                )
                self.vllm_output_queue[uuid] = queue.Queue()
                self.vllm_output_event[uuid] = threading.Event()
            out_tokens = []
            while True:
                self.vllm_output_event[uuid].wait(timeout=0.01)
                self.vllm_output_event[uuid].clear()
                with self.lock:
                    if self.vllm_output_queue[uuid].empty() is True:
                        request_outputs: List[RequestOutput] = self.vllm.step()
                        for request_output in request_outputs:
                            top_ids = list(request_output.outputs[0].token_ids)[-1]
                            self.vllm_output_queue[request_output.request_id].put(
                                top_ids
                            )
                            if request_output.request_id in self.vllm_output_event:
                                self.vllm_output_event[request_output.request_id].set()
                if self.vllm_output_queue[uuid].empty() is False:
                    top_ids = self.vllm_output_queue[uuid].get()
                    if top_ids in self.stop_token_ids:
                        break
                    # in stream mode, yield token one by one
                    yield top_ids
                    out_tokens.append(top_ids)
                    if len(out_tokens) == max_len:
                        break
            with self.lock:
                self.vllm_output_queue.pop(uuid)
                self.vllm_output_event.pop(uuid, None)
        else:
            out_tokens = []
            cache = None
            for i in range(max_len):
                y_pred, cache = self.llm.forward_one_step(
                    lm_input,
                    masks=torch.tril(
                        torch.ones(
                            (1, lm_input.shape[1], lm_input.shape[1]),
                            device=lm_input.device,
                        )
                    ).to(torch.bool),
                    cache=cache,
                )
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                top_ids = self.sampling_ids(
                    logp.squeeze(dim=0),
                    out_tokens,
                    sampling,
                    ignore_eos=True if i < min_len else False,
                )
                if top_ids in self.stop_token_ids:
                    break
                # in stream mode, yield token one by one
                yield top_ids
                out_tokens.append(top_ids)
                lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    @torch.inference_mode()
    def inference_bistream(
        self,
        text: Generator,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = prompt_text.device
        # 1. prepare input
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=prompt_text.dtype
            ).to(device)
        lm_input = torch.concat([sos_emb], dim=1)

        # 2. iterate text
        out_tokens = []
        cache = None
        # NOTE init prompt_text as text_cache as it is basically impossible prompt_speech_token/prompt_text < 15/5
        text_cache = self.llm.model.model.embed_tokens(prompt_text)
        next_fill_index = (
            int(prompt_speech_token.shape[1] / self.mix_ratio[1]) + 1
        ) * self.mix_ratio[1] - prompt_speech_token.shape[1]
        for this_text in text:
            text_cache = torch.concat(
                [text_cache, self.llm.model.model.embed_tokens(this_text)], dim=1
            )
            # prompt_speech_token_emb not empty, try append to lm_input
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text, lm_input_speech = (
                        text_cache[:, : self.mix_ratio[0]],
                        prompt_speech_token_emb[:, : self.mix_ratio[1]],
                    )
                    logging.info(
                        "append {} text token {} speech token".format(
                            lm_input_text.size(1), lm_input_speech.size(1)
                        )
                    )
                    lm_input = torch.concat(
                        [lm_input, lm_input_text, lm_input_speech], dim=1
                    )
                    text_cache, prompt_speech_token_emb = (
                        text_cache[:, self.mix_ratio[0] :],
                        prompt_speech_token_emb[:, self.mix_ratio[1] :],
                    )
                else:
                    logging.info("not enough text token to decode, wait for more")
                    break
            # no prompt_speech_token_emb remain, can decode some speech token
            if prompt_speech_token_emb.size(1) == 0:
                if (len(out_tokens) != 0 and out_tokens[-1] == self.fill_token) or (
                    len(out_tokens) == 0 and lm_input.size(1) == 1
                ):
                    logging.info("get fill token, need to append more text token")
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, : self.mix_ratio[0]]
                        logging.info(
                            "append {} text token".format(lm_input_text.size(1))
                        )
                        if len(out_tokens) != 0 and out_tokens[-1] == self.fill_token:
                            lm_input = lm_input_text
                        else:
                            lm_input = torch.concat([lm_input, lm_input_text], dim=1)
                        text_cache = text_cache[:, self.mix_ratio[0] :]
                    else:
                        logging.info("not enough text token to decode, wait for more")
                        continue
                while True:
                    seq_len = (
                        lm_input.shape[1]
                        if cache is None
                        else lm_input.shape[1] + cache[0][0].size(2)
                    )
                    y_pred, cache = self.llm.forward_one_step(
                        lm_input,
                        masks=torch.tril(
                            torch.ones((1, seq_len, seq_len), device=lm_input.device)
                        ).to(torch.bool),
                        cache=cache,
                    )
                    logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                    if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                        top_ids = self.fill_token
                        next_fill_index += self.mix_ratio[1] + 1
                    else:
                        top_ids = self.sampling_ids(
                            logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True
                        )
                    if top_ids == self.fill_token:
                        next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1
                        logging.info(
                            "fill_token index {} next fill_token index {}".format(
                                len(out_tokens), next_fill_index
                            )
                        )
                    out_tokens.append(top_ids)
                    if top_ids >= self.speech_token_size:
                        if top_ids == self.fill_token:
                            break
                        else:
                            raise ValueError("should not get token {}".format(top_ids))
                    yield top_ids
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        # 3. final decode
        lm_input = torch.concat([lm_input, text_cache, task_id_emb], dim=1)
        logging.info("no more text token, decode until met eos")
        while True:
            seq_len = (
                lm_input.shape[1]
                if cache is None
                else lm_input.shape[1] + cache[0][0].size(2)
            )
            y_pred, cache = self.llm.forward_one_step(
                lm_input,
                masks=torch.tril(
                    torch.ones((1, seq_len, seq_len), device=lm_input.device)
                ).to(torch.bool),
                cache=cache,
            )
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=False
            )
            out_tokens.append(top_ids)
            if top_ids >= self.speech_token_size:
                if top_ids == self.eos_token:
                    break
                else:
                    raise ValueError("should not get token {}".format(top_ids))
            # in stream mode, yield token one by one
            yield top_ids
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    @torch.inference_mode()
    def inference_bistream_vllm(
        self,
        text: Generator,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        """vLLM-based bistream inference with prefix caching support for CosyVoice3.

        This method submits independent requests to vLLM and collects incremental results.
        It leverages vLLM's KV cache to accelerate inference when text is streamed in chunks.
        """
        from vllm import SamplingParams, RequestOutput
        import uuid as uuid_lib

        device = prompt_text.device

        # Timing for first token (will be set when first vLLM request is submitted)
        inference_start_time = None
        first_token_logged = False

        # Prepare embeddings for vllm (using speech_embedding for special tokens in CosyVoice3)
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=prompt_text.dtype
            ).to(device)

        # Initialize state
        out_tokens = []
        text_cache_emb = self.llm.model.model.embed_tokens(prompt_text)
        next_fill_index = (
            int(prompt_speech_token.shape[1] / self.mix_ratio[1]) + 1
        ) * self.mix_ratio[1] - prompt_speech_token.shape[1]

        # Build prefix embeds (will be reused via vLLM's prefix caching)
        prefix_embeds = sos_emb.squeeze(0)  # [1, llm_input_size]
        # current_request_id = 0 # Unused

        for this_text in text:
            # Accumulate text embeddings
            this_text_emb = self.llm.model.model.embed_tokens(this_text)
            text_cache_emb = torch.concat([text_cache_emb, this_text_emb], dim=1)

            # Consume prompt speech tokens (interleaved text-speech mixing)
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache_emb.size(1) >= self.mix_ratio[0]:
                    text_chunk = text_cache_emb[:, : self.mix_ratio[0]]
                    speech_chunk = prompt_speech_token_emb[:, : self.mix_ratio[1]]
                    prefix_embeds = torch.concat(
                        [prefix_embeds, text_chunk.squeeze(0), speech_chunk.squeeze(0)],
                        dim=0,
                    )
                    text_cache_emb = text_cache_emb[:, self.mix_ratio[0] :]
                    prompt_speech_token_emb = prompt_speech_token_emb[
                        :, self.mix_ratio[1] :
                    ]
                else:
                    break

            # When all prompt speech tokens consumed, start generating
            if prompt_speech_token_emb.size(1) == 0:
                # Check if we need more text tokens
                need_text = (
                    len(out_tokens) != 0 and out_tokens[-1] == self.fill_token
                ) or (len(out_tokens) == 0)

                if need_text:
                    if text_cache_emb.size(1) >= self.mix_ratio[0]:
                        text_chunk = text_cache_emb[:, : self.mix_ratio[0]]
                        prefix_embeds = torch.concat(
                            [prefix_embeds, text_chunk.squeeze(0)], dim=0
                        )
                        text_cache_emb = text_cache_emb[:, self.mix_ratio[0] :]
                    else:
                        continue

                # Submit vLLM request with current prefix
                lm_input = prefix_embeds.unsqueeze(0)  # [1, seq_len, hidden_size]

                # Add speech token embeddings for continuation if we have generated tokens
                if len(out_tokens) > 0 and out_tokens[-1] != self.fill_token:
                    last_speech_emb = self.speech_embedding.weight[
                        out_tokens[-1]
                    ].reshape(1, 1, -1)
                    lm_input = torch.concat([lm_input, last_speech_emb], dim=1)

                # Calculate how many tokens to generate until next fill
                tokens_until_fill = (
                    next_fill_index - len(out_tokens)
                    if next_fill_index > len(out_tokens)
                    else self.mix_ratio[1]
                )

                # Use vLLM with stop at fill_token position
                sampling_params = SamplingParams(
                    top_k=sampling,
                    stop_token_ids=self.stop_token_ids,
                    max_tokens=tokens_until_fill,
                    detokenize=False,
                )

                request_id = str(uuid_lib.uuid4())

                # Start timing just before submitting the first vLLM request
                if inference_start_time is None:
                    inference_start_time = time.time()

                with self.lock:
                    self.vllm.add_request(
                        request_id,
                        {"prompt_embeds": lm_input.squeeze(0).to(torch.bfloat16)},
                        sampling_params,
                    )
                    self.vllm_output_queue[request_id] = queue.Queue()

                # Collect tokens from this request
                generated_count = 0
                while True:
                    with self.lock:
                        if self.vllm_output_queue[request_id].empty():
                            request_outputs: List[RequestOutput] = self.vllm.step()
                            for request_output in request_outputs:
                                if request_output.request_id in self.vllm_output_queue:
                                    top_ids = list(request_output.outputs[0].token_ids)[
                                        -1
                                    ]
                                    self.vllm_output_queue[
                                        request_output.request_id
                                    ].put((top_ids, request_output.finished))

                    if not self.vllm_output_queue[request_id].empty():
                        top_ids, finished = self.vllm_output_queue[request_id].get()

                        if top_ids in self.stop_token_ids:
                            break

                        # Check for fill token position
                        if len(out_tokens) == next_fill_index:
                            out_tokens.append(self.fill_token)
                            next_fill_index = len(out_tokens) + self.mix_ratio[1]
                            break

                        out_tokens.append(top_ids)
                        generated_count += 1

                        # Log timing for first token
                        if not first_token_logged:
                            first_token_time = (
                                time.time() - inference_start_time
                            ) * 1000
                            logging.info(f"vllm: first_token={first_token_time:.1f}ms")
                            first_token_logged = True

                        # Update prefix_embeds for next iteration
                        token_emb = self.speech_embedding.weight[top_ids].reshape(1, -1)
                        prefix_embeds = torch.concat([prefix_embeds, token_emb], dim=0)

                        yield top_ids

                        if finished:
                            break

                    time.sleep(0.001)

                with self.lock:
                    self.vllm_output_queue.pop(request_id, None)

        # Final decode: append remaining text and task_id, then generate until EOS

        final_embeds = torch.concat(
            [prefix_embeds, text_cache_emb.squeeze(0), task_id_emb.squeeze(0)], dim=0
        )

        # Add last generated token if exists
        if len(out_tokens) > 0 and out_tokens[-1] < self.speech_token_size:
            last_speech_emb = self.speech_embedding.weight[out_tokens[-1]].reshape(
                1, -1
            )
            final_embeds = torch.concat([final_embeds, last_speech_emb], dim=0)

        lm_input = final_embeds.unsqueeze(0)

        # Calculate remaining max tokens
        text_len = prompt_text_len.item() + text_cache_emb.size(1)
        max_len = int(text_len * max_token_text_ratio) - len(out_tokens)
        max_len = max(max_len, 100)  # ensure reasonable minimum

        sampling_params = SamplingParams(
            top_k=sampling,
            stop_token_ids=self.stop_token_ids,
            max_tokens=max_len,
            detokenize=False,
        )

        request_id = str(uuid_lib.uuid4())

        with self.lock:
            self.vllm.add_request(
                request_id,
                {"prompt_embeds": lm_input.squeeze(0).to(torch.bfloat16)},
                sampling_params,
            )
            self.vllm_output_queue[request_id] = queue.Queue()

        while True:
            with self.lock:
                if self.vllm_output_queue[request_id].empty():
                    request_outputs: List[RequestOutput] = self.vllm.step()
                    for request_output in request_outputs:
                        if request_output.request_id in self.vllm_output_queue:
                            top_ids = list(request_output.outputs[0].token_ids)[-1]
                            self.vllm_output_queue[request_output.request_id].put(
                                (top_ids, request_output.finished)
                            )

            if not self.vllm_output_queue[request_id].empty():
                top_ids, finished = self.vllm_output_queue[request_id].get()

                if top_ids in self.stop_token_ids:
                    break

                out_tokens.append(top_ids)
                yield top_ids

                if finished or len(out_tokens) >= max_len:
                    break

            time.sleep(0.001)

        with self.lock:
            self.vllm_output_queue.pop(request_id, None)
        from vllm import SamplingParams, RequestOutput
        import uuid as uuid_lib

        device = prompt_text.device

        # 1. prepare input embeddings
        # CosyVoice3LM uses speech_embedding for sos/task_id, Qwen2LM uses llm_embedding
        if hasattr(self, "llm_embedding"):
            sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
            task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        else:
            sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
            task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=prompt_text.dtype
            ).to(device)

        # Initialize accumulated embeddings with sos
        accumulated_emb = sos_emb.clone()

        # 2. iterate text
        out_tokens = []
        text_cache = self.llm.model.model.embed_tokens(prompt_text)
        next_fill_index = (
            int(prompt_speech_token.shape[1] / self.mix_ratio[1]) + 1
        ) * self.mix_ratio[1] - prompt_speech_token.shape[1]

        request_uuid = None

        for this_text in text:
            text_cache = torch.concat(
                [text_cache, self.llm.model.model.embed_tokens(this_text)], dim=1
            )

            # prompt_speech_token_emb not empty, try append to accumulated_emb
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text = text_cache[:, : self.mix_ratio[0]]
                    lm_input_speech = prompt_speech_token_emb[:, : self.mix_ratio[1]]
                    logging.info(
                        "append {} text token {} speech token".format(
                            lm_input_text.size(1), lm_input_speech.size(1)
                        )
                    )
                    accumulated_emb = torch.concat(
                        [accumulated_emb, lm_input_text, lm_input_speech], dim=1
                    )
                    text_cache = text_cache[:, self.mix_ratio[0] :]
                    prompt_speech_token_emb = prompt_speech_token_emb[
                        :, self.mix_ratio[1] :
                    ]
                else:
                    logging.info("not enough text token to decode, wait for more")
                    break

            # no prompt_speech_token_emb remain, can decode some speech token
            if prompt_speech_token_emb.size(1) == 0:
                if (len(out_tokens) != 0 and out_tokens[-1] == self.fill_token) or (
                    len(out_tokens) == 0 and accumulated_emb.size(1) == 1
                ):
                    logging.info("get fill token, need to append more text token")
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, : self.mix_ratio[0]]
                        logging.info(
                            "append {} text token".format(lm_input_text.size(1))
                        )
                        accumulated_emb = torch.concat(
                            [accumulated_emb, lm_input_text], dim=1
                        )
                        text_cache = text_cache[:, self.mix_ratio[0] :]
                    else:
                        logging.info("not enough text token to decode, wait for more")
                        continue

                # Generate speech tokens using VLLM
                while True:
                    # Build full prompt: accumulated_emb + generated speech tokens
                    if len(out_tokens) > 0:
                        generated_emb = torch.stack(
                            [self.speech_embedding.weight[t] for t in out_tokens], dim=0
                        ).unsqueeze(0)
                        full_prompt_emb = torch.concat(
                            [accumulated_emb, generated_emb], dim=1
                        )
                    else:
                        full_prompt_emb = accumulated_emb

                    # Create new VLLM request
                    if request_uuid is not None:
                        with self.lock:
                            self.vllm.abort_request(request_uuid)
                            self.vllm_output_queue.pop(request_uuid, None)
                            self.vllm_output_event.pop(request_uuid, None)

                    request_uuid = str(uuid_lib.uuid1())
                    sampling_params = SamplingParams(
                        top_k=sampling,
                        stop_token_ids=self.stop_token_ids,
                        max_tokens=self.mix_ratio[1] + 5,
                    )

                    with self.lock:
                        self.vllm.add_request(
                            request_uuid,
                            {
                                "prompt_embeds": full_prompt_emb.squeeze(0)
                                .to(torch.bfloat16)
                                .to(device)
                            },
                            sampling_params,
                        )
                        self.vllm_output_queue[request_uuid] = queue.Queue()
                        self.vllm_output_event[request_uuid] = threading.Event()

                    # Step VLLM until we get a token
                    got_token = False
                    while not got_token:
                        self.vllm_output_event[request_uuid].wait(timeout=0.01)
                        self.vllm_output_event[request_uuid].clear()

                        with self.lock:
                            if self.vllm_output_queue[request_uuid].empty():
                                request_outputs: List[RequestOutput] = self.vllm.step()
                                for request_output in request_outputs:
                                    top_ids = list(request_output.outputs[0].token_ids)[
                                        -1
                                    ]
                                    if (
                                        request_output.request_id
                                        in self.vllm_output_queue
                                    ):
                                        self.vllm_output_queue[
                                            request_output.request_id
                                        ].put(top_ids)
                                        if (
                                            request_output.request_id
                                            in self.vllm_output_event
                                        ):
                                            self.vllm_output_event[
                                                request_output.request_id
                                            ].set()

                        if not self.vllm_output_queue[request_uuid].empty():
                            top_ids = self.vllm_output_queue[request_uuid].get()

                            # Force fill_token at expected positions
                            if (
                                next_fill_index != -1
                                and len(out_tokens) == next_fill_index
                            ):
                                top_ids = self.fill_token
                                next_fill_index += self.mix_ratio[1] + 1

                            if top_ids == self.fill_token:
                                next_fill_index = (
                                    len(out_tokens) + self.mix_ratio[1] + 1
                                )
                                logging.info(
                                    "fill_token index {} next fill_token index {}".format(
                                        len(out_tokens), next_fill_index
                                    )
                                )

                            out_tokens.append(top_ids)
                            if top_ids >= self.speech_token_size:
                                if top_ids == self.fill_token:
                                    got_token = True  # Break to get more text
                                else:
                                    raise ValueError(
                                        "should not get token {}".format(top_ids)
                                    )
                            else:
                                yield top_ids
                                got_token = True

                    # Check if we got fill_token - break to get more text
                    if len(out_tokens) > 0 and out_tokens[-1] == self.fill_token:
                        break

        # 3. final decode - no more text, decode until eos
        if request_uuid is not None:
            with self.lock:
                self.vllm.abort_request(request_uuid)
                self.vllm_output_queue.pop(request_uuid, None)
                self.vllm_output_event.pop(request_uuid, None)

        # Build final prompt with remaining text and task_id
        if len(out_tokens) > 0:
            generated_emb = torch.stack(
                [self.speech_embedding.weight[t] for t in out_tokens], dim=0
            ).unsqueeze(0)
            full_prompt_emb = torch.concat(
                [accumulated_emb, generated_emb, text_cache, task_id_emb], dim=1
            )
        else:
            full_prompt_emb = torch.concat(
                [accumulated_emb, text_cache, task_id_emb], dim=1
            )

        request_uuid = str(uuid_lib.uuid1())
        sampling_params = SamplingParams(
            top_k=sampling,
            stop_token_ids=self.stop_token_ids,
            max_tokens=1000,
        )

        with self.lock:
            self.vllm.add_request(
                request_uuid,
                {
                    "prompt_embeds": full_prompt_emb.squeeze(0)
                    .to(torch.bfloat16)
                    .to(device)
                },
                sampling_params,
            )
            self.vllm_output_queue[request_uuid] = queue.Queue()
            self.vllm_output_event[request_uuid] = threading.Event()

        logging.info("no more text token, decode until met eos")
        while True:
            self.vllm_output_event[request_uuid].wait(timeout=0.01)
            self.vllm_output_event[request_uuid].clear()

            with self.lock:
                if self.vllm_output_queue[request_uuid].empty():
                    request_outputs: List[RequestOutput] = self.vllm.step()
                    for request_output in request_outputs:
                        top_ids = list(request_output.outputs[0].token_ids)[-1]
                        if request_output.request_id in self.vllm_output_queue:
                            self.vllm_output_queue[request_output.request_id].put(
                                top_ids
                            )
                            if request_output.request_id in self.vllm_output_event:
                                self.vllm_output_event[request_output.request_id].set()

            if not self.vllm_output_queue[request_uuid].empty():
                top_ids = self.vllm_output_queue[request_uuid].get()
                out_tokens.append(top_ids)
                if top_ids >= self.speech_token_size:
                    if top_ids == self.eos_token:
                        break
                    else:
                        raise ValueError("should not get token {}".format(top_ids))
                yield top_ids

        with self.lock:
            self.vllm_output_queue.pop(request_uuid, None)
            self.vllm_output_event.pop(request_uuid, None)


class CosyVoice3LM(Qwen2LM):
    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        llm: torch.nn.Module,
        sampling: Callable,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        mix_ratio: List[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos = speech_token_size + 0
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.fill_token = speech_token_size + 3

        self.llm = llm
        self.llm_decoder = nn.Linear(
            llm_output_size, speech_token_size + 200, bias=False
        )
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 200,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(
            speech_token_size + 200, llm_input_size
        )

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(200)]
        self.vllm_output_queue = {}
        self.vllm_output_event = {}

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)
        # NOTE should append instruct_token to sequence, not implemented yet
        instruct_token = batch["instruct_token"].to(device)
        instruct_token_len = batch["instruct_token_len"].to(device)

        # 1. encode text_token
        text_token_emb = self.llm.model.model.embed_tokens(text_token)

        # 3. sos and task_id
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 2. encode speech_token
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(
            sos_emb,
            text_token,
            text_token_emb,
            text_token_len,
            task_id_emb,
            speech_token,
            speech_token_emb,
            speech_token_len,
        )
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target.to(device))
        acc = th_accuracy(
            logits.view(-1, self.speech_token_size + 200),
            lm_target,
            ignore_label=IGNORE_ID,
        )
        return {"loss": loss, "acc": acc}

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = "",
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)

        # 3. concat llm_input
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=text.dtype
            ).to(device)
        lm_input = torch.concat(
            [sos_emb, text, task_id_emb, prompt_speech_token_emb], dim=1
        )

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token

    @torch.inference_mode()
    def inference_bistream(
        self,
        text: Generator,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = prompt_text.device
        # 1. prepare input
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=prompt_text.dtype
            ).to(device)
        lm_input = torch.concat([sos_emb], dim=1)

        # 2. iterate text
        out_tokens = []
        cache = None
        # NOTE init prompt_text as text_cache as it is basically impossible prompt_speech_token/prompt_text < 15/5
        text_cache = self.llm.model.model.embed_tokens(prompt_text)
        next_fill_index = (
            int(prompt_speech_token.shape[1] / self.mix_ratio[1]) + 1
        ) * self.mix_ratio[1] - prompt_speech_token.shape[1]
        for this_text in text:
            text_cache = torch.concat(
                [text_cache, self.llm.model.model.embed_tokens(this_text)], dim=1
            )
            # prompt_speech_token_emb not empty, try append to lm_input
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text, lm_input_speech = (
                        text_cache[:, : self.mix_ratio[0]],
                        prompt_speech_token_emb[:, : self.mix_ratio[1]],
                    )
                    logging.info(
                        "append {} text token {} speech token".format(
                            lm_input_text.size(1), lm_input_speech.size(1)
                        )
                    )
                    lm_input = torch.concat(
                        [lm_input, lm_input_text, lm_input_speech], dim=1
                    )
                    text_cache, prompt_speech_token_emb = (
                        text_cache[:, self.mix_ratio[0] :],
                        prompt_speech_token_emb[:, self.mix_ratio[1] :],
                    )
                else:
                    logging.info("not enough text token to decode, wait for more")
                    break
            # no prompt_speech_token_emb remain, can decode some speech token
            if prompt_speech_token_emb.size(1) == 0:
                if (len(out_tokens) != 0 and out_tokens[-1] == self.fill_token) or (
                    len(out_tokens) == 0 and lm_input.size(1) == 1
                ):
                    logging.info("get fill token, need to append more text token")
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, : self.mix_ratio[0]]
                        logging.info(
                            "append {} text token".format(lm_input_text.size(1))
                        )
                        if len(out_tokens) != 0 and out_tokens[-1] == self.fill_token:
                            lm_input = lm_input_text
                        else:
                            lm_input = torch.concat([lm_input, lm_input_text], dim=1)
                        text_cache = text_cache[:, self.mix_ratio[0] :]
                    else:
                        logging.info("not enough text token to decode, wait for more")
                        continue
                while True:
                    seq_len = (
                        lm_input.shape[1]
                        if cache is None
                        else lm_input.shape[1] + cache[0][0].size(2)
                    )
                    y_pred, cache = self.llm.forward_one_step(
                        lm_input,
                        masks=torch.tril(
                            torch.ones((1, seq_len, seq_len), device=lm_input.device)
                        ).to(torch.bool),
                        cache=cache,
                    )
                    logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                    if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                        top_ids = self.fill_token
                        next_fill_index += self.mix_ratio[1] + 1
                    else:
                        top_ids = self.sampling_ids(
                            logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True
                        )
                    if top_ids == self.fill_token:
                        next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1
                        logging.info(
                            "fill_token index {} next fill_token index {}".format(
                                len(out_tokens), next_fill_index
                            )
                        )
                    out_tokens.append(top_ids)
                    if top_ids >= self.speech_token_size:
                        if top_ids == self.fill_token:
                            break
                        else:
                            raise ValueError("should not get token {}".format(top_ids))
                    yield top_ids
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        # 3. final decode
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        lm_input = torch.concat([lm_input, text_cache, task_id_emb], dim=1)
        logging.info("no more text token, decode until met eos")
        while True:
            seq_len = (
                lm_input.shape[1]
                if cache is None
                else lm_input.shape[1] + cache[0][0].size(2)
            )
            y_pred, cache = self.llm.forward_one_step(
                lm_input,
                masks=torch.tril(
                    torch.ones((1, seq_len, seq_len), device=lm_input.device)
                ).to(torch.bool),
                cache=cache,
            )
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=False
            )
            out_tokens.append(top_ids)
            if top_ids >= self.speech_token_size:
                if top_ids == self.eos_token:
                    break
                else:
                    raise ValueError("should not get token {}".format(top_ids))
            # in stream mode, yield token one by one
            yield top_ids
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
