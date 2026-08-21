"""
Persistent structured store for any CSV / XLSX upload.

Each file becomes a table with:
- original rows
- inferred schema (column names, dtypes, sample values)

This is independent of embeddings and is what aggregation
questions (count / sum / avg / min / max) must use.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import DATA_DIR, TABLE_STORE_DIR

logger = logging.getLogger(__name__)

INTERNAL_COLUMNS = {
    "__source_file",
    "__document_id",
    "__sheet_name",
    "__row_number",
}


def make_document_id(filename: str) -> str:
    filename = filename.lower()
    filename = re.sub(r"[^a-z0-9.]+", "_", filename).strip("_")
    return f"uploaded_{filename.replace('.', '_')}"


class TableStore:

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or TABLE_STORE_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def upsert_dataframe(
        self,
        df: pd.DataFrame,
        document_id: str,
        document_name: str,
        source_type: str,
    ) -> dict:
        if df is None or df.empty:
            raise ValueError("Cannot store an empty table.")

        stored = df.copy()
        stored["__document_id"] = document_id
        stored["__source_file"] = document_name
        if "__row_number" not in stored.columns:
            stored["__row_number"] = range(1, len(stored) + 1)

        table_dir = self.root / document_id
        table_dir.mkdir(parents=True, exist_ok=True)

        data_path = table_dir / "data.jsonl"
        schema_path = table_dir / "schema.json"

        stored.to_json(
            data_path,
            orient="records",
            lines=True,
            date_format="iso",
        )

        schema = self._build_schema(
            stored,
            document_id=document_id,
            document_name=document_name,
            source_type=source_type,
        )

        schema_path.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "Stored table | id=%s | rows=%d | columns=%d",
            document_id,
            len(stored),
            len(stored.columns),
        )

        return schema

    def upsert_from_file(
        self,
        file_path: Path,
        filename: str,
        document_id: Optional[str] = None,
    ) -> Optional[dict]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        document_id = document_id or make_document_id(filename)

        if suffix == ".csv":
            df = pd.read_csv(path)
            source_type = "csv"
        elif suffix in {".xlsx", ".xls"}:
            df = self._load_excel(path)
            source_type = "xlsx"
        else:
            return None

        if df.empty:
            return None

        return self.upsert_dataframe(
            df=df,
            document_id=document_id,
            document_name=filename,
            source_type=source_type,
        )

    def delete(self, document_id: str) -> None:
        table_dir = self.root / document_id
        if not table_dir.exists():
            return
        for child in table_dir.iterdir():
            child.unlink()
        table_dir.rmdir()

    def list_schemas(self) -> List[dict]:
        schemas = []
        for schema_path in sorted(self.root.glob("*/schema.json")):
            try:
                schemas.append(
                    json.loads(schema_path.read_text(encoding="utf-8"))
                )
            except Exception:
                logger.exception("Failed to read schema %s", schema_path)
        return self._dedupe_schemas(schemas)

    def load_dataframe(self, document_id: str) -> pd.DataFrame:
        data_path = self.root / document_id / "data.jsonl"
        if not data_path.exists():
            return pd.DataFrame()
        return pd.read_json(data_path, lines=True)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        tables = {}
        for schema in self.list_schemas():
            document_id = schema["document_id"]
            df = self.load_dataframe(document_id)
            if not df.empty:
                tables[document_id] = df
        return tables

    def sync_from_data_dir(self) -> None:
        if not DATA_DIR.exists():
            return

        for path in sorted(DATA_DIR.iterdir()):
            if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
                continue
            try:
                self.upsert_from_file(path, path.name)
            except Exception:
                logger.exception("Failed to sync table from %s", path)

    def _load_excel(self, path: Path) -> pd.DataFrame:
        frames = []
        workbook = pd.ExcelFile(path)
        for sheet_name in workbook.sheet_names:
            sheet = pd.read_excel(path, sheet_name=sheet_name)
            if sheet.empty:
                continue
            sheet["__sheet_name"] = str(sheet_name)
            frames.append(sheet)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _build_schema(
        self,
        df: pd.DataFrame,
        document_id: str,
        document_name: str,
        source_type: str,
    ) -> dict:
        columns = []
        for name in df.columns:
            series = df[name]
            unique_texts = []
            for value in series.dropna().astype(str).unique():
                text = str(value).strip()
                if text:
                    unique_texts.append(text)
            sample_limit = 80 if len(unique_texts) <= 80 else 40
            sample_values = unique_texts[:sample_limit]

            columns.append(
                {
                    "name": str(name),
                    "dtype": str(series.dtype),
                    "unique_count": int(series.nunique(dropna=True)),
                    "sample_values": sample_values,
                    "internal": str(name) in INTERNAL_COLUMNS,
                }
            )

        return {
            "document_id": document_id,
            "document_name": document_name,
            "source_type": source_type,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "row_count": int(len(df)),
            "columns": columns,
        }

    def _dedupe_schemas(self, schemas: List[dict]) -> List[dict]:
        """
        If the same column set was indexed twice (CSV + XLSX copy),
        keep the newest table so counts are not doubled.
        """
        grouped: Dict[tuple, dict] = {}
        for schema in schemas:
            fingerprint = tuple(
                col["name"]
                for col in schema.get("columns", [])
                if col["name"] not in INTERNAL_COLUMNS
            )
            current = grouped.get(fingerprint)
            if current is None:
                grouped[fingerprint] = schema
                continue
            if schema.get("indexed_at", "") >= current.get("indexed_at", ""):
                grouped[fingerprint] = schema
        return list(grouped.values())
