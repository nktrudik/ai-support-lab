from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from ai_support_lab.tickets.enums import Category, Priority, TicketStatus


class Customer(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    tier: Literal["free", "pro", "enterprise"] = "free"
    country: str = Field(default="DE", min_length=2, max_length=2)


class TicketCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=5, max_length=10000)
    customer: Customer
    product: str = Field(min_length=1, max_length=80)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("product")
    @classmethod
    def normalize_product(cls, value: str) -> str:
        # Нормализация до записи предотвращает разные фильтры для "Cloud" и "cloud".
        return value.casefold()


class TicketUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str | None = Field(default=None, min_length=3, max_length=200)
    description: str | None = Field(default=None, min_length=5, max_length=10000)
    priority: Priority | None = None
    metadata: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def reject_empty_or_null_patch(self) -> Self:
        # Отсутствующее поле значит «не менять». Явный null запрещён: иначе легко
        # обнулить обязательную колонку или молча получить неожиданную семантику PATCH.
        if not self.model_fields_set or any(
            getattr(self, k) is None for k in self.model_fields_set
        ):
            raise ValueError("Provide at least one non-null field; explicit null is not supported")
        return self


class ClassificationResponse(BaseModel):
    category: Category
    score: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model: str


class PredictionResponse(ClassificationResponse):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime


class TicketResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    id: int
    title: str
    description: str
    customer: Customer
    product: str
    # DeclarativeBase использует имя metadata. Alias сохраняет чистый публичный
    # JSON-контракт, не заставляя ORM переопределять служебный атрибут SQLAlchemy.
    metadata: dict[str, JsonValue] = Field(validation_alias="metadata_json")
    priority: Priority
    status: TicketStatus
    resolution: str | None
    version: int
    created_at: datetime
    updated_at: datetime
    predictions: list[PredictionResponse] = Field(default_factory=list)


class TicketListResponse(BaseModel):
    items: list[TicketResponse]
    total: int
    limit: int
    offset: int


class TicketFilters(BaseModel):
    product: str | None = None
    customer: str | None = None
    priority: Priority | None = None
    status: TicketStatus | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ResolveRequest(BaseModel):
    resolution: str = Field(min_length=5, max_length=4000)
    expected_version: int = Field(ge=1)
