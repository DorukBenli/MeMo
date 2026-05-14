from typing import Optional, Sequence

import torch
from torch import Tensor

from MeMoPyTorch.modelling_memo import MeMo
from MeMoPyTorch.modelling_memo_layer import CompositionOp
from MeMoPyTorch.modelling_memo_exception import MeMoException

from .semantic_store import SemanticStore


class DualTraceMeMo(MeMo):
    """MeMo extended with a MeMo-native semantic memory C_sem.

    Everything from the original MeMo is preserved:
      - C_epi == the CMM of the LAST MeMo layer (`self.layers[-1].CMM`)
      - All MeMo write/read/forget ops work exactly as before.

    New in this version:
      - Padding-aware semantic consolidation.
      - PAD target tokens are not written into C_sem.
      - Context spans containing PAD tokens are not written into C_sem.
      - Semantic forgetting uses the same mask, so it subtracts only valid
        semantic associations.
    """

    def __init__(
        self,
        inner_dim: int,
        num_of_heads: int,
        num_of_layers: int,
        chunk_length: int,
        num_embeddings: int,
        padding_idx: int = 0,
        device: Optional[str] = None,
        alpha_gen: float = 1,
        compositionOp: CompositionOp = CompositionOp.Prod,
        context_layer_k: int = 2,
        blend_lambda: float = 0.3,
    ):
        super().__init__(
            inner_dim,
            num_of_heads,
            num_of_layers,
            chunk_length,
            num_embeddings,
            padding_idx=padding_idx,
            device=device,
            alpha_gen=alpha_gen,
            compositionOp=compositionOp,
        )

        # Some MeMo versions do not keep padding_idx as an exposed attribute,
        # so store it explicitly for our padding-aware semantic mask.
        self.padding_idx = padding_idx

        if context_layer_k < 1 or context_layer_k > num_of_layers:
            raise MeMoException(
                f"context_layer_k={context_layer_k} must be in [1, {num_of_layers}]"
            )
        if self.h ** context_layer_k > self.chunk_length:
            raise MeMoException(
                f"context window h^k={self.h ** context_layer_k} exceeds "
                f"chunk_length={self.chunk_length}"
            )

        self.context_layer_k = context_layer_k
        self.blend_lambda = blend_lambda

        self.C_sem = SemanticStore(self.d, device=self.device)
        self.to(self.device)

    # ------------------------------------------------------------------
    # Section 2 -- write operations
    # ------------------------------------------------------------------

    def memorize(self, input_sequence_ids, labels_ids):
        """Original MeMo memorize + semantic consolidation."""
        super().memorize(input_sequence_ids, labels_ids)
        self.consolidate(input_sequence_ids, labels_ids)

    def consolidate(self, input_sequence_ids, labels_ids, scale: float = 1.0) -> None:
        """Padding-aware semantic consolidation.

        Writes only valid (context, token) pairs into C_sem:
          - target token must not be PAD
          - original context span must not contain PAD
        """
        contexts, tokens, valid_mask = self._layer_k_consolidation_inputs(
            input_sequence_ids,
            labels_ids,
            return_mask=True,
        )

        contexts = contexts[valid_mask]
        tokens = tokens[valid_mask]

        if contexts.numel() == 0:
            return

        self.C_sem.add_pairs(contexts=contexts, tokens=tokens, scale=scale)

    def teach_association(
        self, token_a_id: int, token_b_id: int, weight: float = 1.0
    ) -> None:
        a_vec = self.encoder.weight[token_a_id]
        b_vec = self.encoder.weight[token_b_id]
        self.C_sem.add_association(a_vec, b_vec, scale=weight)

    # ------------------------------------------------------------------
    # Section 3 -- read operations
    # ------------------------------------------------------------------

    def retrieve_episodic(self, input_sequence_ids):
        return self.retrieve(input_sequence_ids)

    def retrieve_semantic(self, input_sequence_ids):
        context_q = self._layer_k_context_query(input_sequence_ids)
        y_hat = self.C_sem(context_q)
        return self.encoder.decode(y_hat)

    def retrieve_blended(
        self,
        input_sequence_ids,
        blend_lambda: Optional[float] = None,
    ):
        if blend_lambda is None:
            blend_lambda = self.blend_lambda

        encoding_for_last_layer, context_q = self._dual_retrieve_encodings(
            input_sequence_ids
        )
        y_epi = self.layers[self.l - 1].directly_retrieve(encoding_for_last_layer)
        y_sem = self.C_sem(context_q)
        y_hat = (1 - blend_lambda) * y_epi + blend_lambda * y_sem
        return self.encoder.decode(y_hat)

    def retrieve_hopfield(
        self,
        input_sequence_ids,
        iterations: int = 3,
        mode: str = "normalize",
    ):
        context_q = self._layer_k_context_query(input_sequence_ids)
        y_hat = self.C_sem.hopfield_retrieve(
            context_q, iterations=iterations, mode=mode
        )
        return self.encoder.decode(y_hat)

    def retrieve_hopfield_blended(
        self,
        input_sequence_ids,
        blend_lambda: Optional[float] = None,
        iterations: int = 3,
        mode: str = "normalize",
    ):
        if blend_lambda is None:
            blend_lambda = self.blend_lambda

        encoding_for_last_layer, context_q = self._dual_retrieve_encodings(
            input_sequence_ids
        )
        y_epi = self.layers[self.l - 1].directly_retrieve(encoding_for_last_layer)
        y_sem = self.C_sem.hopfield_retrieve(
            context_q, iterations=iterations, mode=mode
        )
        y_hat = (1 - blend_lambda) * y_epi + blend_lambda * y_sem
        return self.encoder.decode(y_hat)

    def retrieve_compositional(self, queries_ids_list: Sequence):
        if len(queries_ids_list) == 0:
            raise MeMoException("retrieve_compositional needs at least one query")

        y_hat = None
        for q_ids in queries_ids_list:
            q = self._layer_k_context_query(q_ids)
            proj = self.C_sem(q)
            y_hat = proj if y_hat is None else y_hat * proj
        return self.encoder.decode(y_hat)

    def retrieve_compositional_from_tokens(self, token_ids: Sequence[int]):
        if len(token_ids) == 0:
            raise MeMoException(
                "retrieve_compositional_from_tokens needs at least one token id"
            )

        y_hat = None
        for tid in token_ids:
            t_vec = self.encoder.weight[tid].unsqueeze(0)
            proj = self.C_sem(t_vec)
            y_hat = proj if y_hat is None else y_hat * proj
        return self.encoder.decode(y_hat)

    def semantic_profile(self, token_id: int) -> Tensor:
        t_vec = self.encoder.weight[token_id]
        return self.C_sem.semantic_profile(t_vec)

    def token_semantic_similarity(self, token_a_id: int, token_b_id: int) -> Tensor:
        a_vec = self.encoder.weight[token_a_id]
        b_vec = self.encoder.weight[token_b_id]
        return self.C_sem.profile_similarity(a_vec, b_vec)

    # ------------------------------------------------------------------
    # Section 4 -- forgetting
    # ------------------------------------------------------------------

    def forget(
        self,
        input_sequence_ids,
        labels_ids,
        beta_e: float = 1.0,
        beta_s: float = 1.0,
        completely: bool = True,
    ) -> None:
        """Generalised forget with independent episodic/semantic strength.

        beta_e controls original MeMo forgetting.
        beta_s controls semantic associative forgetting.
        """
        if beta_e > 0:
            if beta_e == 1.0:
                super().forget(input_sequence_ids, labels_ids, completely=completely)
            else:
                self._scaled_episodic_forget(
                    input_sequence_ids, labels_ids, scale=beta_e
                )

        if beta_s > 0:
            contexts, tokens, valid_mask = self._layer_k_consolidation_inputs(
                input_sequence_ids,
                labels_ids,
                return_mask=True,
            )

            contexts = contexts[valid_mask]
            tokens = tokens[valid_mask]

            if contexts.numel() > 0:
                self.C_sem.sub_pairs(contexts=contexts, tokens=tokens, scale=beta_s)

    def forget_association(
        self,
        token_a_id: int,
        token_b_id: int,
        weight: Optional[float] = None,
    ) -> None:
        a_vec = self.encoder.weight[token_a_id]
        b_vec = self.encoder.weight[token_b_id]

        if weight is None:
            denom = (a_vec @ a_vec).item() * (b_vec @ b_vec).item()
            raw = self.C_sem.association_strength(a_vec, b_vec).item()
            weight = raw / denom if denom > 0 else 0.0

        self.C_sem.sub_association(a_vec, b_vec, scale=weight)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _layer_k_consolidation_inputs(
        self,
        input_sequence_ids,
        labels_ids,
        return_mask: bool = False,
    ):
        """Replay MeMo's per-layer reindexing up to layer k.

        Returns:
          sequence_encoding: encoded context vectors, shape (batch, blocks, d)
          output_symbols: encoded target-token vectors, shape (batch, blocks, d)

        If return_mask=True, also returns:
          valid_mask: boolean mask, shape (batch, blocks)

        valid_mask removes:
          1. PAD target tokens
          2. contexts that contain PAD tokens in their original input span
        """
        raw_input_ids = input_sequence_ids
        raw_label_ids = labels_ids

        input_sequence = self.encoder.encode(input_sequence_ids)
        output_symbols = self.encoder.encode(labels_ids)

        (_, current_length, _) = input_sequence.shape
        if current_length != self.chunk_length:
            raise MeMoException(
                f"Expected chunk length {self.chunk_length}, got {current_length}"
            )

        target_idx = self.context_layer_k - 1

        for layer_level in range(self.l):
            if self.h ** (layer_level + 1) >= current_length + 1:
                raise MeMoException(
                    f"Cannot reach layer k={self.context_layer_k}: chunk too short."
                )

            layer_output_idxs = [
                i - self.h ** layer_level
                for i in range(self.h ** (layer_level + 1), current_length + 1)
            ]
            output_symbols = output_symbols[:, layer_output_idxs]

            input_index = [
                list(range(i - self.h ** (layer_level + 1), i, self.h ** layer_level))
                for i in range(self.h ** (layer_level + 1), current_length + 1)
            ]
            input_sequence = input_sequence[:, input_index]

            (_, blocks, h, dim) = input_sequence.shape
            sequence_encoding, _ = self.layers[layer_level].get_projections(
                input_sequence, blocks, h, dim
            )

            if layer_level == target_idx:
                if return_mask:
                    # Target token must not be PAD.
                    target_ids = raw_label_ids[:, layer_output_idxs]
                    target_valid = target_ids != self.padding_idx

                    # Original context span must not contain PAD.
                    # Shape: (batch, blocks, h)
                    context_ids = raw_input_ids[:, input_index]
                    context_valid = (context_ids != self.padding_idx).all(dim=-1)

                    valid_mask = target_valid & context_valid
                    return sequence_encoding, output_symbols, valid_mask

                return sequence_encoding, output_symbols

            input_sequence = sequence_encoding

        raise MeMoException(
            f"Did not reach layer k={self.context_layer_k}; check num_of_layers."
        )

    def _layer_k_context_query(self, input_sequence_ids) -> Tensor:
        """Run k MeMo layers in retrieve mode and return the most recent
        block's sequence encoding -- the query for C_sem.
        Returns shape (batch, d).
        """
        input_sequence = self.encoder(input_sequence_ids)
        (batch_size, length, _) = input_sequence.shape

        if length != self.max_len:
            raise MeMoException(
                f"Expected input of length {self.max_len}, got {length}"
            )

        current = length
        for layer_level in range(self.context_layer_k):
            current = current // self.h
            input_sequence = input_sequence.reshape(
                (batch_size, current, self.h, self.d)
            )
            input_sequence, _ = self.layers[layer_level].retrieve(input_sequence)

        return input_sequence[:, -1]

    def _dual_retrieve_encodings(self, input_sequence_ids):
        """Single forward pass through ALL l layers that returns BOTH:
            - encoding_for_last_layer: the accumulated query for C_epi
            - context_q:               the layer-k query for C_sem
        """
        input_sequence = self.encoder(input_sequence_ids)
        (batch_size, length, _) = input_sequence.shape

        if length != self.max_len:
            raise MeMoException(
                f"Expected input of length {self.max_len}, got {length}"
            )

        encoding_for_last_layer = torch.zeros(
            (batch_size, self.d), device=self.device
        )
        target_idx = self.context_layer_k - 1
        context_q = None
        current = length

        for layer_level in range(self.l):
            current = current // self.h
            input_sequence = input_sequence.reshape(
                (batch_size, current, self.h, self.d)
            )
            input_sequence, seq_enc_last = self.layers[layer_level].retrieve(
                input_sequence
            )
            encoding_for_last_layer = encoding_for_last_layer + seq_enc_last

            if layer_level == target_idx:
                context_q = input_sequence[:, -1]

        if context_q is None:
            raise MeMoException(
                f"Could not capture layer-k context query (k={self.context_layer_k})."
            )

        return encoding_for_last_layer, context_q

    def _scaled_episodic_forget(
        self,
        input_sequence_ids,
        labels_ids,
        scale: float,
    ) -> None:
        """Partial forget of C_epi by `scale in (0, 1)`.

        Only the LAST layer's CMM (== C_epi) is updated; the per-layer
        intermediate CMMs are left untouched, since the spec treats C_epi as a
        single matrix. For full MeMo-standard forgetting of intermediate
        stores, use beta_e=1.0, which delegates to MeMo.forget.
        """
        input_sequence = self.encoder.encode(input_sequence_ids)
        output_symbols = self.encoder.encode(labels_ids)

        (_, current_length, _) = input_sequence.shape
        if current_length != self.chunk_length:
            raise MeMoException(
                f"Expected chunk length {self.chunk_length}, got {current_length}"
            )

        last_layer = self.layers[self.l - 1]

        for layer_level in range(self.l):
            if self.h ** (layer_level + 1) >= current_length + 1:
                break

            layer_output_idxs = [
                i - self.h ** layer_level
                for i in range(self.h ** (layer_level + 1), current_length + 1)
            ]
            output_symbols = output_symbols[:, layer_output_idxs]

            input_index = [
                list(range(i - self.h ** (layer_level + 1), i, self.h ** layer_level))
                for i in range(self.h ** (layer_level + 1), current_length + 1)
            ]
            input_sequence = input_sequence[:, input_index]

            (_, blocks, h, dim) = input_sequence.shape
            sequence_encoding, seq_enc_per_token = self.layers[
                layer_level
            ].get_projections(input_sequence, blocks, h, dim)

            seq_enc_plus_out = torch.matmul(
                seq_enc_per_token.transpose(-2, -1), output_symbols
            )

            if seq_enc_plus_out.dim() == 3:
                seq_enc_plus_out = seq_enc_plus_out.sum(dim=0)

            with torch.no_grad():
                last_layer.CMM.weight.sub_(seq_enc_plus_out, alpha=scale)

            input_sequence = sequence_encoding
