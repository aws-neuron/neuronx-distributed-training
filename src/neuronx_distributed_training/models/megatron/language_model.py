# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer based language model."""

import torch
import torch_xla.core.xla_model as xm
from apex.transformer.enums import AttnMaskType
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)
from neuronx_distributed.parallel_layers import layers, mappings, random

from .transformer import ParallelTransformer


def get_language_model(
    hidden_size,
    ffn_hidden_size,
    num_layers,
    max_position_embeddings,
    num_tokentypes,
    add_pooler,
    vocab_size,
    num_attention_heads,
    encoder_attn_mask_type,
    apply_query_key_layer_scaling=True,
    kv_channels=None,
    init_method=None,
    scaled_init_method=None,
    add_decoder=False,
    decoder_attn_mask_type=AttnMaskType.causal,
    init_method_std=0.02,
    resume_from_checkpoint=False,
    use_cpu_initialization=False,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    ffn_dropout=0.0,
    precision=16,
    fp32_residual_connection=False,
    activations_checkpoint_method=None,
    activations_checkpoint_num_layers=1,
    normalization="layernorm",
    layernorm_epsilon=1e-5,
    bias_activation_fusion=True,
    masked_softmax_fusion=True,
    activation="gelu",
    headscale=False,
    transformer_block_type="pre_ln",
    normalize_attention_scores=True,
    position_embedding_type="learned_absolute",
    attention_type="multihead",
    share_embeddings_and_output_weights=True,
    rotary_percentage=1.0,
    multi_query_attention=False,
    bias_dropout_add_fusion=True,
    bias=True,
    gradient_accumulation_fusion=False,
    persist_layer_norm=False,
    openai_gelu=False,
    megatron_legacy=False,
    activations_checkpoint_granularity=None,
    activations_checkpoint_layers_per_pipeline=None,
    sequence_parallel=False,
    reduce_amax=True,
    use_emha=False,
    position_interpolation_factor=1.0,
    position_freq_base=10000,
    position_abf_factor=1,
    num_kv_heads=None,
    sliding_window=None,
    num_moe_experts=1,
    moe_frequency=1,
    moe_dropout=0.0,
    moe_capacity_factor=1.0,
    moe_sinkhorn_iterations=30,
    moe_sinkhorn_tol=None,
    moe_routing_algorithm="top_k",
    moe_router_activation="sigmoid",
    moe_top_k=1,
    output_router_logits=False,
    expert_model_parallel_size=1,
    router_aux_loss_coef=0.02,
    normalize_top_k_affinities=True,
):
    """Build language model and return along with the key to save."""

    if kv_channels is None:
        assert (
            hidden_size % num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads if kv_channels is None"
        kv_channels = hidden_size // num_attention_heads
    if num_kv_heads is not None:
        assert num_attention_heads % num_kv_heads == 0, "number of query heads should be divisible by kv heads"
    if init_method is None:
        init_method = init_method_normal(init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

    # Language model.
    language_model = TransformerLanguageModel(
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        encoder_attn_mask_type=encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        kv_channels=kv_channels,
        ffn_hidden_size=ffn_hidden_size,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        resume_from_checkpoint=resume_from_checkpoint,
        use_cpu_initialization=use_cpu_initialization,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        ffn_dropout=ffn_dropout,
        precision=precision,
        fp32_residual_connection=fp32_residual_connection,
        activations_checkpoint_method=activations_checkpoint_method,
        activations_checkpoint_num_layers=activations_checkpoint_num_layers,
        normalization=normalization,
        layernorm_epsilon=layernorm_epsilon,
        bias_activation_fusion=bias_activation_fusion,
        bias_dropout_add_fusion=bias_dropout_add_fusion,
        bias=bias,
        rotary_percentage=rotary_percentage,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        masked_softmax_fusion=masked_softmax_fusion,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
        activation=activation,
        headscale=headscale,
        transformer_block_type=transformer_block_type,
        normalize_attention_scores=normalize_attention_scores,
        position_embedding_type=position_embedding_type,
        multi_query_attention=multi_query_attention,
        persist_layer_norm=persist_layer_norm,
        openai_gelu=openai_gelu,
        megatron_legacy=megatron_legacy,
        activations_checkpoint_granularity=activations_checkpoint_granularity,
        activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
        sequence_parallel=sequence_parallel,
        reduce_amax=reduce_amax,
        use_emha=use_emha,
        position_interpolation_factor=position_interpolation_factor,
        position_freq_base=position_freq_base,
        position_abf_factor=position_abf_factor,
        num_kv_heads=num_kv_heads,
        sliding_window=sliding_window,
        num_moe_experts=num_moe_experts,
        moe_frequency=moe_frequency,
        moe_dropout=moe_dropout,
        moe_capacity_factor=moe_capacity_factor,
        moe_sinkhorn_iterations=moe_sinkhorn_iterations,
        moe_sinkhorn_tol=moe_sinkhorn_tol,
        moe_routing_algorithm=moe_routing_algorithm,
        moe_router_activation=moe_router_activation,
        moe_top_k=moe_top_k,
        output_router_logits=output_router_logits,
        expert_model_parallel_size=expert_model_parallel_size,
        router_aux_loss_coef=router_aux_loss_coef,
        normalize_top_k_affinities=normalize_top_k_affinities,
    )

    # key used for checkpoints.
    language_model_key = "language_model"

    return language_model, language_model_key


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
        resume_from_checkpoint: whether the training was resuming from a checkpoint. If so, then no need to initialize the weights
        use_cpu_initialization: whether to initialize the weights in CPU
        position_embedding_type: position embedding type determines whether we instantiate a learnable position embedding table.
    """

    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        resume_from_checkpoint=False,
        use_cpu_initialization=False,
        fp32_residual_connection=False,
        sequence_parallel=False,
        position_embedding_type="learned_absolute",
        transpose_batch_sequence=True,
    ):
        super(Embedding, self).__init__()
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.position_embedding_type = position_embedding_type
        self.transpose_batch_sequence = transpose_batch_sequence
        import torch_xla.core.xla_model as xm

        # Word embeddings (parallel).
        self.word_embeddings = layers.ParallelEmbedding(
            vocab_size,
            self.hidden_size,
            init_method=self.init_method,
            # resume_from_checkpoint=resume_from_checkpoint,
            # use_cpu_initialization=use_cpu_initialization,
            device=None if use_cpu_initialization else xm.xla_device(),
        )
        self._word_embeddings_key = "word_embeddings"

        if self.position_embedding_type == "learned_absolute":
            # Position embedding (serial).
            self.position_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size)
            self._position_embeddings_key = "position_embeddings"
            # Initialize the position embeddings.
            self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = "tokentype_embeddings"
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size)
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.fp32_residual_connection = fp32_residual_connection
        self.sequence_parallel = sequence_parallel

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        if self.position_embedding_type == "learned_absolute":
            self.position_embeddings.weight.data.fill_(0)
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception("tokentype embeddings is already initialized")
        if torch.distributed.get_rank() == 0:
            print("adding embedding for {} tokentypes".format(num_tokentypes), flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes, self.hidden_size)
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        if self.position_embedding_type == "learned_absolute":
            assert position_ids is not None
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings
        if token_type_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(token_type_ids)
        else:
            assert self.tokentype_embeddings is None

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        if self.transpose_batch_sequence:
            embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = mappings.scatter_to_sequence_parallel_region(embeddings)
            with random.get_xla_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings


class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        encoder_attn_mask_type,
        vocab_size,
        max_position_embeddings,
        hidden_size,
        ffn_hidden_size,
        num_layers,
        num_tokentypes,
        num_attention_heads,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        add_decoder=False,
        decoder_attn_mask_type=AttnMaskType.causal,
        add_pooler=False,
        resume_from_checkpoint=False,
        use_cpu_initialization=False,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        normalization="layernorm",
        layernorm_epsilon=1e-5,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        bias=True,
        masked_softmax_fusion=True,
        activation="gelu",
        headscale=False,
        transformer_block_type="pre_ln",
        normalize_attention_scores=True,
        position_embedding_type="learned_absolute",
        rotary_percentage=1.0,
        multi_query_attention=False,
        share_embeddings_and_output_weights=True,
        gradient_accumulation_fusion=False,
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
        megatron_legacy=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_layers_per_pipeline=None,
        sequence_parallel=False,
        reduce_amax=True,
        use_emha=False,
        position_interpolation_factor=1.0,
        position_freq_base=10000,
        position_abf_factor=1,
        num_kv_heads=None,
        sliding_window=None,
        num_moe_experts=1,
        moe_frequency=1,
        moe_dropout=0.0,
        moe_capacity_factor=1.0,
        moe_sinkhorn_iterations=30,
        moe_sinkhorn_tol=None,
        moe_routing_algorithm="top_k",
        moe_router_activation="sigmoid",
        moe_top_k=1,
        output_router_logits=False,
        expert_model_parallel_size=1,
        router_aux_loss_coef=0.02,
        normalize_top_k_affinities=True,
    ):
        super(TransformerLanguageModel, self).__init__(share_token_embeddings=share_embeddings_and_output_weights)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.hidden_dropout = hidden_dropout
        self.output_layer_init_method = output_layer_init_method
        self.position_embedding_type = position_embedding_type
        self.position_interpolation_factor = position_interpolation_factor
        self.position_freq_base = position_freq_base
        self.position_abf_factor = position_abf_factor
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.sequence_parallel = sequence_parallel
        self.rotary_percentage = rotary_percentage
        self.output_router_logits = output_router_logits
        assert 0 < rotary_percentage <= 1
        assert not add_pooler, "Pooler is not supported"

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), "hidden_size must be divisible by num_attention_heads if kv_channels is None"
            kv_channels = hidden_size // num_attention_heads

        # Embeddings.
        self.embedding = Embedding(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_position_embeddings,
            init_method=self.init_method,
            num_tokentypes=self.num_tokentypes,
            resume_from_checkpoint=resume_from_checkpoint,
            use_cpu_initialization=use_cpu_initialization,
            embedding_dropout_prob=self.hidden_dropout,
            sequence_parallel=sequence_parallel,
            position_embedding_type=position_embedding_type,
            fp32_residual_connection=fp32_residual_connection,
        )
        self._embedding_key = "embedding"
        self.encoder = ParallelTransformer(
            layer_type=LayerType.encoder,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            self_attn_mask_type=self.encoder_attn_mask_type,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            normalization=normalization,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            resume_from_checkpoint=resume_from_checkpoint,
            use_cpu_initialization=use_cpu_initialization,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            bias=bias,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            activation=activation,
            headscale=headscale,
            transformer_block_type=transformer_block_type,
            normalize_attention_scores=normalize_attention_scores,
            multi_query_attention=multi_query_attention,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            megatron_legacy=megatron_legacy,
            sequence_parallel=sequence_parallel,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
            reduce_amax=reduce_amax,
            use_emha=use_emha,
            position_embedding_type=self.position_embedding_type,
            position_interpolation_factor=self.position_interpolation_factor,
            position_freq_base=self.position_freq_base,
            position_abf_factor=self.position_abf_factor,
            max_position_embeddings=self.max_position_embeddings,
            rotary_percentage=self.rotary_percentage,
            num_kv_heads=num_kv_heads,
            sliding_window=sliding_window,
            num_moe_experts=num_moe_experts,
            moe_frequency=moe_frequency,
            moe_dropout=moe_dropout,
            moe_capacity_factor=moe_capacity_factor,
            moe_sinkhorn_iterations=moe_sinkhorn_iterations,
            moe_sinkhorn_tol=moe_sinkhorn_tol,
            moe_routing_algorithm=moe_routing_algorithm,
            moe_router_activation=moe_router_activation,
            moe_top_k=moe_top_k,
            output_router_logits=output_router_logits,
            expert_model_parallel_size=expert_model_parallel_size,
            router_aux_loss_coef=router_aux_loss_coef,
            normalize_top_k_affinities=normalize_top_k_affinities,
        )
        self._encoder_key = "encoder"

        self.output_layer = layers.ColumnParallelLinear(
            self.hidden_size,
            self.vocab_size,
            bias=False,  # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
            init_method=self.init_method,
            device=None if use_cpu_initialization else xm.xla_device(),
            gather_output=False,
            sequence_parallel_enabled=sequence_parallel,
        )
        self._output_layer_key = "output_layer"

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        self.encoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        enc_input_ids,
        enc_position_ids,
        enc_attn_mask,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attn_mask=None,
        enc_dec_attn_mask=None,
        token_type_ids=None,
        layer_past=None,
        get_key_value=None,
        pooling_sequence_index=0,
        enc_hidden_states=None,
        output_enc_hidden_only=None,
        encoder_input=None,
        set_inference_key_value_memory=None,
        inference_max_sequence_len=None,
        checkpoint_activations_all_layers=None,
    ):
        # Embeddings.
        if encoder_input is None:
            encoder_input = self.embedding(enc_input_ids, enc_position_ids, token_type_ids=token_type_ids)
        else:
            pass
        rotary_pos_emb = None
        if enc_hidden_states is None:
            encoder_output = self.encoder(
                encoder_input,
                enc_attn_mask,
                layer_past=layer_past,
                get_key_value=get_key_value,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
                rotary_pos_emb=(rotary_pos_emb, None, None)
                if rotary_pos_emb is not None
                else None,  # This assumes that this being used as a GPT/BERT model only (no cross-attention)
            )
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.output_router_logits:
            combined_output = encoder_output
            encoder_output = combined_output[0]
            router_logits = combined_output[1]

        encoder_output = self.output_layer(encoder_output)
        return (encoder_output, router_logits) if self.output_router_logits else encoder_output
