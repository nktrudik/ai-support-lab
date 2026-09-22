from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ai_support_lab.tickets.enums import Category, Priority, TicketStatus


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (Index("ix_tickets_product_created", "product", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    customer: Mapped[dict[str, Any]] = mapped_column(JSON)
    customer_name: Mapped[str] = mapped_column(String(120), index=True)
    product: Mapped[str] = mapped_column(String(80))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    priority: Mapped[Priority] = mapped_column(
        Enum(Priority, native_enum=False), default=Priority.MEDIUM
    )
    status: Mapped[TicketStatus] = mapped_column(
        Enum(TicketStatus, native_enum=False), default=TicketStatus.OPEN
    )
    resolution: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    predictions: Mapped[list["TicketPrediction"]] = relationship(
        back_populates="ticket",
        cascade="all, delete-orphan",
        lazy="raise",
        order_by="TicketPrediction.id",
    )
    agent_runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="ticket",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    # Optimistic locking защищает обычные изменения, а агент дополнительно сверяет
    # версию снимка: за время HTTP-вызова LLM пользователь мог изменить обращение.
    __mapper_args__ = {"version_id_col": version}


class TicketPrediction(Base):
    __tablename__ = "ticket_predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"), index=True)
    category: Mapped[Category] = mapped_column(Enum(Category, native_enum=False))
    score: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict[str, float]] = mapped_column(JSON)
    model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    ticket: Mapped[Ticket] = relationship(back_populates="predictions", lazy="raise")


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"), index=True)
    ticket_version: Mapped[int]
    backend: Mapped[str] = mapped_column(String(40))
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    ticket: Mapped[Ticket] = relationship(back_populates="agent_runs", lazy="raise")
