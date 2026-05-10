import logging
from typing import TypedDict

from app.db.embeddings import clear_workspace_embeddings, ensure_schema_table, upsert_table_embedding
from app.db.sql import fetch_table_schema

logger = logging.getLogger(__name__)


class TableInfo(TypedDict):
    table_name: str
    column_names: list[str]
    description: str


_WORKSPACE_TABLE_META: dict[str, list[TableInfo]] = {
    "logistics-schema": [
        {
            "table_name": "shipments",
            "column_names": ["id", "origin_hub", "destination_hub", "status", "shipped_at", "delivered_at"],
            "description": "Core shipments table tracking all packages with origin/destination hubs and timestamps",
        },
        {
            "table_name": "shipments_events",
            "column_names": ["id", "shipment_id", "event_type", "hub", "occurred_at"],
            "description": "Event log for shipments — scans, handling, transfers, delivery attempts",
        },
        {
            "table_name": "lcr",
            "column_names": ["id", "origin", "destination", "rate", "currency", "effective_from"],
            "description": "Land carrier rates — shipping cost between hub pairs by currency",
        },
        {
            "table_name": "lcr_routes",
            "column_names": ["id", "route_name", "origin", "destination", "carrier"],
            "description": "Route definitions linking origin and destination hubs with carrier info",
        },
        {
            "table_name": "hubs",
            "column_names": ["id", "code", "city", "state", "region", "capacity"],
            "description": "All hub locations with codes, cities, states and capacity",
        },
    ],
}


async def index_workspace(workspace: str) -> None:
    await ensure_schema_table()
    tables_meta = _WORKSPACE_TABLE_META.get(workspace, [])
    if not tables_meta:
        logger.warning("No table metadata defined for workspace: %s", workspace)
        return

    await clear_workspace_embeddings(workspace)
    for table_meta in tables_meta:
        await upsert_table_embedding(
            workspace=workspace,
            table_name=table_meta["table_name"],
            column_names=table_meta["column_names"],
            description=table_meta["description"],
        )
        logger.info("Indexed table %s in workspace %s", table_meta["table_name"], workspace)


async def index_all_workspaces() -> None:
    for workspace in _WORKSPACE_TABLE_META:
        await index_workspace(workspace)
