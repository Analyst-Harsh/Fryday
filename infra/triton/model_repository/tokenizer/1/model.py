"""Triton Python backend: text → input_ids + attention_mask for the e5 embedder.

Uses the plain `tokenizers` library with e5's tokenizer.json (copied next to this file,
so the server never downloads anything). Padding and truncation match HF's
AutoTokenizer(padding=True, truncation=True, max_length=512): checked 2026-11-15.
"""

from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils  # provided by Triton at run time
from tokenizers import Tokenizer

MAX_LEN = 512  # BERT's position-embedding limit


class TritonPythonModel:
    def initialize(self, args: dict[str, str]) -> None:
        """Runs once when Triton loads the model: load the tokenizer."""
        version_dir = Path(args["model_repository"]) / args["model_version"]
        self.tok = Tokenizer.from_file(str(version_dir / "tokenizer.json"))
        pad_id = self.tok.token_to_id("<pad>")
        self.tok.enable_padding(pad_id=pad_id, pad_token="<pad>")  # noqa: S106 -- padding token, not a credential
        self.tok.enable_truncation(max_length=MAX_LEN)

    def execute(self, requests: list) -> list:
        """Runs per batch of requests; returns exactly one response per request, in order."""
        responses = []
        for request in requests:
            # Client input is untrusted: a bad request (e.g. invalid UTF-8) gets its own
            # error response instead of failing every request in this batch.
            try:
                # TYPE_STRING arrives as a numpy array of bytes objects, shape [batch, 1].
                text = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
                encs = self.tok.encode_batch([t.decode("utf-8") for t in text.reshape(-1)])
                ids = np.array([e.ids for e in encs], dtype=np.int64)
                mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("input_ids", ids),
                            pb_utils.Tensor("attention_mask", mask),
                        ]
                    )
                )
            except UnicodeDecodeError as e:  # the client's fault → HTTP 400, not 500
                err = pb_utils.TritonError(f"tokenizer: {e}", pb_utils.TritonError.INVALID_ARG)
                responses.append(pb_utils.InferenceResponse(error=err))
            except Exception as e:  # anything else is our bug → 500
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(f"tokenizer: {e}"))
                )
        return responses
