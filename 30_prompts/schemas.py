# schemas.py
# Canonical Pydantic schemas for all pipeline API calls.
# Import from here in every pipeline script — do not redefine inline.

from typing import Literal
from pydantic import BaseModel, Field


# %% Coworker filter — stage 30
class CoworkerFilter(BaseModel):
    has_coworkers: bool


# %% Text generation — stage 40
class TextsOutput(BaseModel):
    text1: str
    text2: str
    text3: str
    text4: str
    text5: str
    def as_list(self) -> list[str]: return [self.text1, self.text2, self.text3, self.text4, self.text5]


# %% BPNS completion — stage 50
# Field naming: q_ = questionnaire, aut/com/rel = need acronym, _N = item number
class BPNSCompletion(BaseModel):
    q_aut_1: int = Field(ge=1, le=5)
    q_aut_2: int = Field(ge=1, le=5)
    q_aut_3: int = Field(ge=1, le=5)
    q_aut_4: int = Field(ge=1, le=5)
    q_com_1: int = Field(ge=1, le=5)
    q_com_2: int = Field(ge=1, le=5)
    q_com_3: int = Field(ge=1, le=5)
    q_com_4: int = Field(ge=1, le=5)
    q_rel_1: int = Field(ge=1, le=5)
    q_rel_2: int = Field(ge=1, le=5)
    q_rel_3: int = Field(ge=1, le=5)
    q_rel_4: int = Field(ge=1, le=5)


# %% LLM scoring — stage 60
# Field naming: llm_ prefix distinguishes scorer output from questionnaire items
class LLMScoring(BaseModel):
    llm_aut: int = Field(ge=1, le=5)
    llm_com: int = Field(ge=1, le=5)
    llm_rel: int = Field(ge=1, le=5)
