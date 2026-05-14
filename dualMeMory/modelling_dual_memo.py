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

    The dual-trace contribution is additive and matches the architecture
    described in CLAUDE.md:

      - C_sem is a `SemanticStore` exposing two matrices,
            C_ctx2tok  : context -> token
            C_tok2ctx  : token   -> context
        which are updated together every time a (context, token) pair is
        consolidated. The two-direction storage is what turns MeMo into a
        dual-trace associative memory: paraphrases share contexts, so their
        `semantic_profile(token) = token_vector @ C_tok2ctx` vectors converge
        even though their random token embeddings differ.
      - consolidate(): hook fired automatically after every memorize().
      - teach_association() / forget_association(): direct knowledge edits.
      - retrieve_semantic() / retrieve_blended() / retrieve_compositional().
      - semantic_profile() / token_semantic_similarity(): paraphrase probes.
      - forget() generalised to a (beta_e, beta_s) policy.
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

    # 2.1 Episodic encoding -- unchanged. We override only to chain consolidation.
    def memorize(self, input_sequence_ids, labels_ids):
        super().memorize(input_sequence_ids, labels_ids)
        self.consolidate(input_sequence_ids, labels_ids)

    # 2.2 Consolidation: dual outer-product update of C_ctx2tok and C_tok2ctx
    #     at layer k. Following CLAUDE.md, for every (context_i, token_i)
    #     occurrence we write
    #         C_ctx2tok += outer(context_i, token_i)
    #         C_tok2ctx += outer(token_i, context_i)
    def consolidate(self, input_sequence_ids, labels_ids, scale: float = 1.0) -> None:
        contexts, tokens = self._layer_k_consolidation_inputs(
            input_sequence_ids, labels_ids
        )
        self.C_sem.add_pairs(contexts=contexts, tokens=tokens, scale=scale)

    # 2.3 Direct association injection. `a` is treated as the context cue and
    #     `b` as the predicted token; the symmetric tok2ctx update is added in
    #     SemanticStore.add_association.
    def teach_association(
        self, token_a_id: int, token_b_id: int, weight: float = 1.0
    ) -> None:
        a_vec = self.encoder.weight[token_a_id]
        b_vec = self.encoder.weight[token_b_id]
        self.C_sem.add_association(a_vec, b_vec, scale=weight)

    # ------------------------------------------------------------------
    # Section 3 -- read operations
    # ------------------------------------------------------------------

    # 3.1 Episodic retrieval -- unchanged. Aliased for symmetry of the API.
    def retrieve_episodic(self, input_sequence_ids):
        return self.retrieve(input_sequence_ids)

    # 3.2 Semantic retrieval (queries C_ctx2tok only).
    def retrieve_semantic(self, input_sequence_ids):
        context_q = self._layer_k_context_query(input_sequence_ids)
        y_hat = self.C_sem(context_q)
        return self.encoder.decode(y_hat)

    # 3.3 Blended retrieval -- the main inference operation.
    #     y_hat = (1 - lambda) * y_epi + lambda * y_sem
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

    # 3.4 Hopfield-style attractor retrieval. The query context is iteratively
    #     refined through C_sem so paraphrases collapse onto the same attractor
    #     before the final token read-out.
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

    # 3.5 Compositional inference over multiple input cues.
    def retrieve_compositional(self, queries_ids_list: Sequence):
        if len(queries_ids_list) == 0:
            raise MeMoException("retrieve_compositional needs at least one query")

        y_hat = None
        for q_ids in queries_ids_list:
            q = self._layer_k_context_query(q_ids)
            proj = self.C_sem(q)
            y_hat = proj if y_hat is None else y_hat * proj
        return self.encoder.decode(y_hat)

    # Token-cue variant: each cue is a single token id; query C_ctx2tok
    # directly with the token embedding (matching the teach_association
    # primitive, where the first argument is used as a context cue).
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

    # 3.6 Distributional semantic profile of a single token.
    #     semantic_profile(t) = t_vec @ C_tok2ctx  (CLAUDE.md eq.).
    def semantic_profile(self, token_id: int) -> Tensor:
        t_vec = self.encoder.weight[token_id]
        return self.C_sem.semantic_profile(t_vec)

    # 3.7 Paraphrase-style similarity between two tokens, derived purely from
    #     shared associative contexts (no external encoder).
    def token_semantic_similarity(self, token_a_id: int, token_b_id: int) -> Tensor:
        a_vec = self.encoder.weight[token_a_id]
        b_vec = self.encoder.weight[token_b_id]
        return self.C_sem.profile_similarity(a_vec, b_vec)

    # ------------------------------------------------------------------
    # Section 4 -- forgetting
    # ------------------------------------------------------------------

    # 4.1 / 4.2 Generalised forget with (beta_e, beta_s).
    #     Semantic forgetting weakens BOTH directions of C_sem so that the
    #     dual store stays internally consistent after the edit.
    def forget(
        self,
        input_sequence_ids,
        labels_ids,
        beta_e: float = 1.0,
        beta_s: float = 1.0,
        completely: bool = True,
    ) -> None:
        if beta_e > 0:
            if beta_e == 1.0:
                super().forget(input_sequence_ids, labels_ids, completely=completely)
            else:
                self._scaled_episodic_forget(
                    input_sequence_ids, labels_ids, scale=beta_e
                )

        if beta_s > 0:
            contexts, tokens = self._layer_k_consolidation_inputs(
                input_sequence_ids, labels_ids
            )
            self.C_sem.sub_pairs(contexts=contexts, tokens=tokens, scale=beta_s)

    # 4.3 Targeted semantic forgetting (no episode required).
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

    def _layer_k_consolidation_inputs(self, input_sequence_ids, labels_ids):
        """Replay MeMo's per-layer reindexing up to layer k and return the
        (context_keys, token_values) pairs ready for the C_sem outer-product
        update. Does NOT mutate any CMM.
        """
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
        self, input_sequence_ids, labels_ids, scale: float
    ) -> None:
        """Partial forget of C_epi by `scale in (0, 1)`.

        Only the LAST layer's CMM (== C_epi) is updated; the per-layer
        intermediate CMMs are left untouched, since the spec treats C_epi as a
        single matrix. For full MeMo-standard forgetting of intermediate
        stores, use beta_e=1.0 (which delegates to MeMo.forget).
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
