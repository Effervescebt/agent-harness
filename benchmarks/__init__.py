from .gsm8k import GSM8K
from .hotpotqa import HotpotQA
from .hotpotqa_session import HotpotQASession

BENCHMARKS = {b.name: b for b in (HotpotQA, HotpotQASession, GSM8K)}
