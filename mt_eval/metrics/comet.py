from __future__ import annotations

from comet import download_model, load_from_checkpoint

_model = None

# Patch comet encoders for compatibility with transformers v5.
# transformers v5 removed build_inputs_with_special_tokens from fast tokenizers
# and changed XLMRobertaModel tuple output to drop the pooler when not needed.
def _apply_comet_patches():
    import torch
    from typing import Dict, List
    from comet.encoders import base as _base_mod
    from comet.encoders import xlmr as _xlmr_mod

    def _concat_sequences(self, inputs, return_label_ids=False):
        concat_input_ids = []
        for encoder_input in inputs:
            input_ids = encoder_input["input_ids"]
            input_ids = [
                x.masked_select(x.ne(self.tokenizer.pad_token_id)).tolist()
                for x in input_ids.unbind(dim=0)
            ]
            concat_input_ids.append(input_ids)

        batch_size = len(concat_input_ids[0])
        batch = []
        token_type_ids = []
        for i in range(batch_size):
            lengths = tuple(len(x[i][1:-1]) for x in concat_input_ids)
            special_tokens = 1 + len(inputs) * self.size_separator
            new_sequence = concat_input_ids[0][i]
            token_type_ids.append(
                torch.zeros(len(new_sequence[1:-1]) + 2, dtype=torch.int)
            )
            for j in range(1, len(inputs)):
                seq_a = new_sequence[1:-1]
                seq_b = concat_input_ids[j][i][1:-1]
                seq_a = seq_a.tolist() if hasattr(seq_a, "tolist") else list(seq_a)
                seq_b = seq_b.tolist() if hasattr(seq_b, "tolist") else list(seq_b)
                new_sequence = (
                    [self.tokenizer.bos_token_id]
                    + seq_a
                    + [self.tokenizer.eos_token_id]
                    + seq_b
                    + [self.tokenizer.eos_token_id]
                )
            if sum(lengths) > self.max_positions - special_tokens:
                new_sequence = new_sequence[: self.max_positions]
            batch.append(torch.tensor(new_sequence))

        lengths = [t.shape[0] for t in batch]
        max_len = max(lengths)
        padded = [self.pad_tensor(t, max_len, self.tokenizer.pad_token_id) for t in batch]
        lengths = torch.tensor(lengths, dtype=torch.long)
        padded = torch.stack(padded, dim=0).contiguous()
        attention_mask = torch.arange(max_len)[None, :] < lengths[:, None]
        if return_label_ids:
            label_ids = [self.pad_tensor(t, max_len, -1) for t in inputs[0]["label_ids"]]
            label_ids = torch.stack(label_ids, dim=0).contiguous()
            encoder_input = {
                "input_ids": padded,
                "attention_mask": attention_mask,
                "label_ids": label_ids,
                "mt_offsets": inputs[0]["offsets"],
            }
        else:
            encoder_input = {"input_ids": padded, "attention_mask": attention_mask}
        if self.uses_token_type_ids:
            token_type_ids = [self.pad_tensor(t, max_len, 1) for t in token_type_ids]
            token_type_ids = torch.stack(token_type_ids, dim=0).contiguous()
            encoder_input["token_type_ids"] = token_type_ids
        return encoder_input, lengths, max_len

    def _xlmr_forward(self, input_ids, attention_mask, **kwargs):
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )
        last_hidden_states = model_output[0]
        all_layers = model_output[-1]
        return {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }

    _base_mod.Encoder.concat_sequences = _concat_sequences
    _xlmr_mod.XLMREncoder.forward = _xlmr_forward


_apply_comet_patches()


def _get_model():
    global _model
    if _model is None:
        model_path = download_model("McGill-NLP/ssa-comet-mtl")
        _model = load_from_checkpoint(model_path)
    return _model


def score_ssa_comet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    batch_size: int = 8,
    gpus: int = 1,
) -> tuple[list[float], float]:
    """Run SSA-COMET on a src, hyp, ref triples.

    Returns:
        sentence_scores: per-sample scores (0–1)
        system_score:    corpus-level score (0–1)
    """
    model = _get_model()
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    prediction = model.predict(data, batch_size=batch_size, gpus=gpus)
    return prediction.scores, prediction.system_score
