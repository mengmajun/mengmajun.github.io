+++ 
date = '2025-07-03' 
draft = false 
title = 'KV-cache原理' 
categories = ['大模型', '推理优化'] 
tags = ['KV-cache'] 
+++



## KV Cache 原理详解

![KV-cache图解]()

### 1. 概述

KV Cache（Key-Value Cache）是大型语言模型在**推理阶段**中广泛使用的一种优化技术。它的核心目的是**减少重复计算**，提升生成过程中每一步的效率。

在自回归生成任务中，模型每次只需预测下一个 token，因此我们不需要每次都重新处理整个历史序列。KV Cache 的作用就是**缓存之前所有 token 对应的 Key 和 Value 向量**，从而避免重复计算，加快推理速度。

---

### 2. 核心原理

大模型通常采用**因果解码器结构**，即每个位置的 token 只能关注到它之前的 token，而不能看到未来的 token。

假设当前我们要预测第 $ t $ 个 token，根据注意力机制的定义，第 $ t-1 $ 个 token 的注意力计算是不需要看到第 $ t $ 个 token 的。这说明，在推理过程中，**历史的 Key 和 Value 是不会变化的**，可以被缓存起来供后续步骤复用。

---

### 3. 工作流程

以下是 KV Cache 在推理中的典型工作流程：

1. **初始输入**：给定一个起始序列（例如 prompt），模型会为这个序列中的每个 token 计算对应的 Query (Q)、Key (K)、Value (V) 向量。
   
   - 输出形状通常是：`(batch_size, n_head, seq_len, n_embed)`，其中：
     - `batch_size`：批量大小
     - `n_head`：注意力头的数量
     - `seq_len`：当前输入序列长度
     - `n_embed`：每个 token 的嵌入维度

2. **缓存 K 和 V**：将这些 Key 和 Value 缓存起来，作为后续推理的 KV Cache。

3. **下一步推理**：
   - 输入上一步输出的 token（即 next token）；
   - 模型仅对这个新 token 进行线性变换，得到新的 Q、K、V，`(batch_size, n_head, 1, n_embed)`；
   - 新的 K 和 V 与缓存中的历史 K/V 拼接，形成完整的 K-state 和 V-state；
   - 使用更新后的 K-state 和 V-state 与当前 token 的 Q 计算注意力；
   - 将新生成的 K 和 V 加入缓存，准备下一轮使用。

4. **循环进行**：不断重复上述过程，直到生成完整的目标序列或达到最大长度限制。

---

### 4. 优势与意义

- **显著减少计算量**：每一时刻只需要处理一个 token，其余的 K/V 直接从缓存中读取。
- **节省内存带宽**：避免重复传输和计算历史数据，但是在实践中要注意可能爆内存。
- **加速推理过程**：尤其在长序列生成任务中效果明显。

---

### 5. 代码实现

在实现层面，KV Cache 的管理非常关键：

- 每一层 Transformer 都需要维护自己的 Key 和 Value 缓存；
- 每次推理时，新 token 的 K 和 V 会追加到缓存中；
- 注意力计算时使用的是当前的 Query 和完整的 K-state / V-state；
- 整个缓存结构需支持动态扩展（按 sequence length 维度拼接）。

---

## transformers库 KV-cache源码解读

已GPT2为例，代码文件modeling_gpt2.py

GPT2Attention中实现了核心的KV-cache计算拼接

```python
class GPT2Attention(nn.Module):
    ...
        def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # 计算当前token的QKV向量，hidden_states只是一个token
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        # 使用kv-cache情况下，这里计算的是一个token的q k v，这种情况下也只有一个token输入到了模型
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        # 这里判断是否使用KV-cache
        if layer_past is not None:
            past_key, past_value = layer_past
            # 按照seq_length维度拼接 比如past_key是(1,12,1,64), key_states是(1,12,1,64)，拼接后是(1,12,2,64)
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        if use_cache is True:
            present = (key_states, value_states)
        else:
            present = None

        is_cross_attention = encoder_hidden_states is not None
        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
                # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
                # not necessarily to eager (if mentionned options are provided).
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        # 将KV-cache返回
        return outputs  # a, present, (attentions)
```

GPT2Model遍历每一个transformer block，使用KV-cache计算注意力



```python

class GPT2Model(GPT2PreTrainedModel):
    ...
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    ...
    presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.h)):
            # 取出来每一层的block，每一层block缓存的key value
            block, layer_past = self.h[i], past_key_values[i]

            ...

            # 调用GPT2Attention计算，传入当前层的KV-cache
            outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

```


generate方法中会调用_sample方法，实现每次传入模型序列最后一个token，使用KV-cache进行计算


```python

class GenerationMixin:
    ...
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    ...
            while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            # 使用kv cache情况下，返回序列最后一个token id用于继续生成
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                # 模型输入本次最后一个token
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            # 将生成的最新token追加到输入序列中，用于下一次生成，dim=-1表示在seq_length上拼接
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

```


## KV-cache效果对比

```python

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

model_path = r"D:\project\python\demo\hugging_face_model\gpt2"

# 加载模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

text = "hello tom, "
encoded_input = tokenizer(text, return_tensors="pt")

# 测试开启KV Cache
start_time = time.time()
output_cache = model.generate(
    **encoded_input,
    max_new_tokens=50,
    use_cache=True,  # 开启KV Cache
    pad_token_id=tokenizer.eos_token_id
)
cache_time = time.time() - start_time
decoded_cache = tokenizer.decode(output_cache[0], skip_special_tokens=True)

# 测试关闭KV Cache
start_time = time.time()
output_no_cache = model.generate(
    **encoded_input,
    max_new_tokens=50,
    use_cache=False,  # 关闭KV Cache
    pad_token_id=tokenizer.eos_token_id
)
no_cache_time = time.time() - start_time
decoded_no_cache = tokenizer.decode(output_no_cache[0], skip_special_tokens=True)

# 对比结果
print("=== With KV Cache ===")
print(f"Generation time: {cache_time:.2f}s")
print(f"Generated text: {decoded_cache}\n")

print("=== Without KV Cache ===")
print(f"Generation time: {no_cache_time:.2f}s")
print(f"Generated text: {decoded_no_cache}\n")

# 计算加速比
if no_cache_time > 0:
    speedup = no_cache_time / cache_time
    print(f"KV Cache speedup: {speedup:.1f}x")
else:
    print("Cannot calculate speedup (no_cache_time is 0)")


"""
=== With KV Cache ===
Generation time: 1.26s
Generated text: hello tom, ive been here for a while and i have been here for a long time. i have been here for a long time and i have been here for a long time. i have been here for a long time and i have been here for a long

=== Without KV Cache ===
Generation time: 2.14s
Generated text: hello tom, ive been here for a while and i have been here for a long time. i have been here for a long time and i have been here for a long time. i have been here for a long time and i have been here for a long

KV Cache speedup: 1.7x
"""

```
