"""Pydantic data models for the ConvFinQA dataset (see dataset.md)."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    pre_text: str = Field(description="The text before the table in the document")
    post_text: str = Field(description="The text after the table in the document")
    table: dict[str, dict[str, float | str | int]] = Field(
        description="The table of the document as a dictionary"
    )


class Dialogue(BaseModel):
    conv_questions: list[str] = Field(description="The questions in the conversation dialogue")
    conv_answers: list[str] = Field(description="The answers to each question turn")
    turn_program: list[str] = Field(description="The DSL turn program for each question turn")
    executed_answers: list[float | str] = Field(
        description="The golden program execution results for each question turn"
    )
    qa_split: list[bool] = Field(
        description="Source of each turn: False if from the first FinQA question, True if from the second"
    )


class Features(BaseModel):
    num_dialogue_turns: int
    has_type2_question: bool
    has_duplicate_columns: bool
    has_non_numeric_values: bool


class ConvFinQARecord(BaseModel):
    id: str = Field(description="The id of the record")
    doc: Document
    dialogue: Dialogue
    features: Features
