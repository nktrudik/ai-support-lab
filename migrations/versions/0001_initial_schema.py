"""Создать tickets, predictions и agent_runs"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "bdd443135f76"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Фиксированная схема миграции не зависит от будущих изменений ORM-классов.
    op.create_table(
        "tickets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("customer", sa.JSON(), nullable=False),
        sa.Column("customer_name", sa.String(length=120), nullable=False),
        sa.Column("product", sa.String(length=80), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column(
            "priority",
            sa.Enum("LOW", "MEDIUM", "HIGH", "CRITICAL", name="priority", native_enum=False),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("OPEN", "RESOLVED", name="ticketstatus", native_enum=False),
            nullable=False,
        ),
        sa.Column("resolution", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_tickets_customer_name"), "tickets", ["customer_name"], unique=False)
    op.create_index(
        "ix_tickets_product_created", "tickets", ["product", "created_at"], unique=False
    )
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("ticket_id", sa.Integer(), nullable=False),
        sa.Column("ticket_version", sa.Integer(), nullable=False),
        sa.Column("backend", sa.String(length=40), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["ticket_id"], ["tickets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_agent_runs_ticket_id"), "agent_runs", ["ticket_id"], unique=False)
    op.create_table(
        "ticket_predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("ticket_id", sa.Integer(), nullable=False),
        sa.Column(
            "category",
            sa.Enum(
                "BILLING",
                "AUTHENTICATION",
                "PERFORMANCE",
                "INTEGRATION",
                "ACCOUNT",
                "BUG",
                "FEATURE_REQUEST",
                name="category",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("probabilities", sa.JSON(), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["ticket_id"], ["tickets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_ticket_predictions_ticket_id"), "ticket_predictions", ["ticket_id"], unique=False
    )


def downgrade() -> None:
    # Фиксированная схема миграции не зависит от будущих изменений ORM-классов.
    op.drop_index(op.f("ix_ticket_predictions_ticket_id"), table_name="ticket_predictions")
    op.drop_table("ticket_predictions")
    op.drop_index(op.f("ix_agent_runs_ticket_id"), table_name="agent_runs")
    op.drop_table("agent_runs")
    op.drop_index("ix_tickets_product_created", table_name="tickets")
    op.drop_index(op.f("ix_tickets_customer_name"), table_name="tickets")
    op.drop_table("tickets")
