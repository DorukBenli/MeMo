import torch
from torch import Tensor
from torch.nn import Module, init
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter


class SemanticStore(Module):
    """MeMo-native semantic memory C_sem, realised as a *pair* of
    correlation-matrix memories:

        C_ctx2tok : context -> token   (used for semantic retrieval)
        C_tok2ctx : token   -> context (used to compute a token's
                                        distributional semantic profile)

    Both matrices have shape (d, d) and are zero-initialised, exactly like
    every other MeMo CMM. They are updated together on every consolidation
    step so that:

        C_ctx2tok += sum_i outer(context_i, token_i)
        C_tok2ctx += sum_i outer(token_i, context_i)

    Each token therefore gradually acquires a *context profile* via

        semantic_profile(token) = token_vector @ C_tok2ctx

    and two tokens become semantically close when they share contexts -- the
    MeMo-native source of paraphrase awareness described in CLAUDE.md. The
    two matrices remain independently editable through the explicit API
    below, which is what enables targeted semantic forgetting / profile
    editing without touching the other direction.

    Storage / retrieval conventions match MeMo's CMM_OUT:

        write:  C += outer(key, value)
        read:   predicted_value = key_query @ C
    """

    def __init__(self, d: int, device=None, dtype=None, init_weights: bool = True):
        super().__init__()
        self.d = d
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ctx2tok = Parameter(
            torch.empty((d, d), **factory_kwargs), requires_grad=False
        )
        self.tok2ctx = Parameter(
            torch.empty((d, d), **factory_kwargs), requires_grad=False
        )
        if init_weights:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.ctx2tok)
        init.zeros_(self.tok2ctx)

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    def forward(self, context_query: Tensor) -> Tensor:
        """Semantic retrieval: context -> token. Shape (..., d) -> (..., d)."""
        return torch.matmul(context_query, self.ctx2tok)

    def semantic_profile(self, token_query: Tensor) -> Tensor:
        """Context profile of a token: token -> context. Shape (..., d) -> (..., d)."""
        return torch.matmul(token_query, self.tok2ctx)

    def association_strength(self, a_vec: Tensor, b_vec: Tensor) -> Tensor:
        """Strength of the ctx-key=a -> tok-value=b association."""
        return a_vec.flatten() @ self.ctx2tok @ b_vec.flatten()

    def profile_similarity(self, a_vec: Tensor, b_vec: Tensor, eps: float = 1e-12) -> Tensor:
        """Cosine similarity between the context profiles of two tokens."""
        a_prof = self.semantic_profile(a_vec.flatten())
        b_prof = self.semantic_profile(b_vec.flatten())
        denom = torch.linalg.norm(a_prof) * torch.linalg.norm(b_prof)
        return (a_prof @ b_prof) / (denom + eps)

    # ------------------------------------------------------------------
    # Write paths -- consolidation of (context, token) pairs
    # ------------------------------------------------------------------

    def _pair_update(self, keys: Tensor, values: Tensor) -> Tensor:
        k = keys.reshape(-1, self.d)
        v = values.reshape(-1, self.d)
        return k.transpose(0, 1) @ v

    def add_pairs(self, contexts: Tensor, tokens: Tensor, scale: float = 1.0) -> None:
        """Symmetric two-direction outer-product update."""
        c2t = self._pair_update(contexts, tokens)
        t2c = self._pair_update(tokens, contexts)
        with torch.no_grad():
            self.ctx2tok.add_(c2t, alpha=scale)
            self.tok2ctx.add_(t2c, alpha=scale)

    def sub_pairs(self, contexts: Tensor, tokens: Tensor, scale: float = 1.0) -> None:
        c2t = self._pair_update(contexts, tokens)
        t2c = self._pair_update(tokens, contexts)
        with torch.no_grad():
            self.ctx2tok.sub_(c2t, alpha=scale)
            self.tok2ctx.sub_(t2c, alpha=scale)

    # ------------------------------------------------------------------
    # Write paths -- direct token-token associations
    # ------------------------------------------------------------------

    def add_association(self, a_vec: Tensor, b_vec: Tensor, scale: float = 1.0) -> None:
        """Inject one (a, b) association: a acts as the context cue, b as the
        predicted token. Both directions are updated symmetrically."""
        a = a_vec.flatten()
        b = b_vec.flatten()
        c2t = torch.outer(a, b)
        t2c = torch.outer(b, a)
        with torch.no_grad():
            self.ctx2tok.add_(c2t, alpha=scale)
            self.tok2ctx.add_(t2c, alpha=scale)

    def sub_association(self, a_vec: Tensor, b_vec: Tensor, scale: float = 1.0) -> None:
        a = a_vec.flatten()
        b = b_vec.flatten()
        c2t = torch.outer(a, b)
        t2c = torch.outer(b, a)
        with torch.no_grad():
            self.ctx2tok.sub_(c2t, alpha=scale)
            self.tok2ctx.sub_(t2c, alpha=scale)

    # ------------------------------------------------------------------
    # Hopfield-style attractor retrieval (CLAUDE.md, Hopfield section)
    #
    # The dual matrices implicitly define a symmetric attractor operator
    #     S_eff = ctx2tok @ tok2ctx = sum_{i,j} <x_i, x_j> c_i c_j^T,
    # which collapses to sum_i c_i c_i^T when token vectors are
    # near-orthogonal (the MeMo random-embedding regime). Iterating
    #     q_{t+1} = phi( q_t @ ctx2tok @ tok2ctx )
    # therefore moves the query toward the nearest learned context attractor.
    # ------------------------------------------------------------------

    def hopfield_step(self, context_query: Tensor, mode: str = "normalize") -> Tensor:
        """One Hopfield round-trip in context space: ctx -> tok -> ctx."""
        tok = torch.matmul(context_query, self.ctx2tok)
        ctx_new = torch.matmul(tok, self.tok2ctx)
        if mode == "normalize":
            return normalize(ctx_new, p=2, dim=-1)
        if mode == "sign":
            return torch.sign(ctx_new)
        if mode == "none":
            return ctx_new
        raise ValueError(f"Unknown hopfield mode: {mode!r}")

    def hopfield_retrieve(
        self,
        context_query: Tensor,
        iterations: int = 3,
        mode: str = "normalize",
    ) -> Tensor:
        """Iterate Hopfield updates, then read out the token prediction."""
        q = context_query
        for _ in range(iterations):
            q = self.hopfield_step(q, mode=mode)
        return torch.matmul(q, self.ctx2tok)

    def hopfield_trajectory(
        self,
        context_query: Tensor,
        iterations: int = 3,
        mode: str = "normalize",
    ) -> Tensor:
        """Return all intermediate states (iterations+1, ..., d) for diagnostics."""
        states = [context_query]
        q = context_query
        for _ in range(iterations):
            q = self.hopfield_step(q, mode=mode)
            states.append(q)
        return torch.stack(states, dim=0)
