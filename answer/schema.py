from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class AnswerCitation:
    chunk_id: str
    doc_title: str
    doc_number: str
    section: str
    article: str
    source: str
    text: str


@dataclass
class VerificationResult:
    passed: bool
    unsupported_claims: List[str] = field(default_factory=list)
    missing_conditions: List[str] = field(default_factory=list)
    citation_errors: List[str] = field(default_factory=list)
    should_refuse: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class AnswerResult:
    answer: str
    status: str = 'answered'
    confidence: str = 'medium'
    citations: List[AnswerCitation] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    missing_conditions: List[str] = field(default_factory=list)
    verification: VerificationResult = field(default_factory=lambda: VerificationResult(True))

    def to_dict(self):
        return asdict(self)
