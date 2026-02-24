# -*- coding: utf-8 -*-
"""
숫자 토큰만 생성하도록 제한하는 LogitsProcessor 유틸.
- 토크나이저 종류(특히 SentencePiece)별로 '1', ' 1', '▁1' 등 단일 토큰 변형을 자동 탐색.
- digits="1234" 같이 전달하면 해당 숫자들만 허용.
"""

from typing import Iterable, Set, List
import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase


def _find_single_token_ids_for_digits(
    tokenizer: PreTrainedTokenizerBase,
    digits: Iterable[str],
) -> Set[int]:
    """
    단일 토큰 디코딩 결과가 (공백 유무 무시 시) 숫자 한 글자와 동일한 모든 vocab id를 수집.
    ex) decode([tid]).strip() == '1'
    """
    target = {d.strip() for d in digits if d and d.strip().isdigit()}
    allow_ids: Set[int] = set()

    # 전체 vocab을 한 번 스캔 (한 번만 하면 됨)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        # 일부 sentencepiece 계열은 get_vocab() 사용
        vocab_size = len(tokenizer.get_vocab())

    # 배치 디코딩 없이 단건 디코딩(가장 안전)
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid], skip_special_tokens=True)
        except Exception:
            continue
        if not s:
            continue
        # 좌우 공백 제거 후 숫자 한 글자만 남는지 확인
        if s.strip() in target:
            allow_ids.add(tid)

    return allow_ids


class AllowedTokensProcessor(LogitsProcessor):
    """
    scores: (batch_size, vocab_size)
    → 허용된 토큰 id만 점수 유지, 나머지는 -inf로 마스킹
    """
    def __init__(self, allowed_token_ids: Iterable[int]):
        ids = list(dict.fromkeys(int(i) for i in allowed_token_ids))
        if len(ids) == 0:
            raise ValueError("허용 토큰 id가 비어 있습니다.")
        self._allowed = torch.tensor(ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: (B, V)
        device = scores.device
        allowed = self._allowed.to(device)
        mask = torch.full_like(scores, float("-inf"))
        # 모든 배치에 동일한 허용 집합 적용
        mask.index_fill_(1, allowed, 0.0)
        return scores + mask


def get_digit_processor(tokenizer: PreTrainedTokenizerBase, digits: str = "1234") -> AllowedTokensProcessor:
    """
    예) digits="12345" → 1~5 중 1토큰만 생성 허용.
    """
    if not digits or any(ch for ch in digits if not ch.isdigit()):
        raise ValueError("digits 인자는 '1234' 같은 숫자 문자열이어야 합니다.")
    allow_ids = _find_single_token_ids_for_digits(tokenizer, digits)
    return AllowedTokensProcessor(allow_ids)