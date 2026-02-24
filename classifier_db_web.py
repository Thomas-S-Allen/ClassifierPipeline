#!/usr/bin/env python3
"""Web UI for inspecting and updating classifier_pipeline records over HTTP.

This keeps the same functional layout as classifier_db_ui.py, but serves a browser UI.
"""

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

import psycopg2
import requests
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from psycopg2.extras import RealDictCursor


ALLOWED_CATEGORIES = [
    "astrophysics",
    "heliophysics",
    "planetary",
    "earthscience",
    "NASA-funded Biophysics",
    "physics",
    "general",
    "Text Garbage",
]

ADS_API_URL = "https://api.adsabs.harvard.edu/v1/search/query"


@dataclass(frozen=True)
class QuerySpec:
    label: str
    needs_run_id: bool = False
    needs_bibcode_term: bool = False


QUERY_SPECS = [
    QuerySpec("Latest records"),
    QuerySpec("Unvalidated records"),
    QuerySpec("Validated records"),
    QuerySpec("By run_id", needs_run_id=True),
    QuerySpec("By bibcode contains", needs_bibcode_term=True),
]
QUERY_SPEC_BY_LABEL = {spec.label: spec for spec in QUERY_SPECS}


class DatabaseClient:
    def __init__(self):
        self.conn = None
        self.metadata_table = None

    def connect(self, *, host: str, port: str, dbname: str, user: str, password: str):
        self.close()
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        self.conn.autocommit = False
        self.metadata_table = self._detect_metadata_table()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            self.metadata_table = None

    def _detect_metadata_table(self):
        sql = """
            SELECT table_name
            FROM (
                SELECT
                    table_name,
                    MAX(CASE WHEN column_name = 'bibcode' THEN 1 ELSE 0 END) AS has_bibcode,
                    MAX(CASE WHEN column_name = 'title' THEN 1 ELSE 0 END) AS has_title,
                    MAX(CASE WHEN column_name = 'abstract' THEN 1 ELSE 0 END) AS has_abstract
                FROM information_schema.columns
                WHERE table_schema = 'public'
                GROUP BY table_name
            ) t
            WHERE has_bibcode = 1 AND has_title = 1 AND has_abstract = 1
            ORDER BY CASE
                WHEN table_name = 'records' THEN 0
                WHEN table_name = 'input_records' THEN 1
                WHEN table_name = 'master_records' THEN 2
                ELSE 10
            END,
            table_name
            LIMIT 1
        """
        with self.conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
        return row[0] if row else None

    def _base_select(self):
        if self.metadata_table:
            metadata_join = f"LEFT JOIN {self.metadata_table} md ON md.bibcode = s.bibcode"
            title_expr = "COALESCE(md.title, '') AS title"
            abstract_expr = "COALESCE(md.abstract, '') AS abstract"
        else:
            metadata_join = ""
            title_expr = "'' AS title"
            abstract_expr = "'' AS abstract"

        return f"""
            SELECT
                s.id AS score_id,
                s.bibcode,
                s.scix_id,
                s.run_id,
                s.scores,
                {title_expr},
                {abstract_expr},
                fc.id AS final_collection_id,
                fc.collection,
                fc.validated,
                ov.override
            FROM scores s
            LEFT JOIN LATERAL (
                SELECT id, collection, validated
                FROM final_collection
                WHERE score_id = s.id
                    OR (score_id IS NULL AND bibcode = s.bibcode)
                ORDER BY
                    CASE WHEN score_id = s.id THEN 0 ELSE 1 END,
                    created DESC
                LIMIT 1
            ) fc ON TRUE
            LEFT JOIN LATERAL (
                SELECT override
                FROM overrides
                WHERE bibcode = s.bibcode
                ORDER BY created DESC
                LIMIT 1
            ) ov ON TRUE
            {metadata_join}
        """

    def run_query(self, *, spec: QuerySpec, run_id: str, bibcode_term: str, limit: int):
        where_clauses = []
        params = []

        if spec.needs_run_id:
            if not run_id.strip():
                raise ValueError("run_id is required for this query.")
            where_clauses.append("s.run_id = %s")
            params.append(int(run_id))

        if spec.needs_bibcode_term:
            if not bibcode_term.strip():
                raise ValueError("Bibcode text is required for this query.")
            where_clauses.append("s.bibcode ILIKE %s")
            params.append(f"%{bibcode_term.strip()}%")

        if spec.label == "Unvalidated records":
            where_clauses.append("COALESCE(fc.validated, FALSE) = FALSE")
        if spec.label == "Validated records":
            where_clauses.append("COALESCE(fc.validated, FALSE) = TRUE")

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        sql = self._base_select() + where_sql + " ORDER BY s.id DESC LIMIT %s"
        params.append(limit)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def update_collection(
        self,
        *,
        final_collection_id,
        score_id,
        bibcode,
        scix_id,
        collection,
        validated,
    ):
        if not self.conn:
            raise RuntimeError("No database connection.")

        with self.conn.cursor() as cur:
            updated = 0
            if score_id:
                cur.execute(
                    """
                    UPDATE final_collection
                    SET collection = %s, validated = %s
                    WHERE score_id = %s
                    """,
                    (collection, validated, score_id),
                )
                updated = cur.rowcount

            if updated == 0 and final_collection_id:
                cur.execute(
                    """
                    UPDATE final_collection
                    SET collection = %s, validated = %s
                    WHERE id = %s
                    """,
                    (collection, validated, final_collection_id),
                )
                updated = cur.rowcount

            if updated == 0:
                cur.execute(
                    """
                    INSERT INTO final_collection (bibcode, score_id, collection, validated)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (bibcode, score_id, collection, validated),
                )

            override_updated = 0
            if scix_id:
                cur.execute(
                    """
                    UPDATE overrides
                    SET override = %s
                    WHERE scix_id = %s
                    """,
                    (collection, scix_id),
                )
                override_updated = cur.rowcount

            if override_updated == 0 and bibcode:
                cur.execute(
                    """
                    UPDATE overrides
                    SET override = %s
                    WHERE bibcode = %s
                    """,
                    (collection, bibcode),
                )
                override_updated = cur.rowcount

            if override_updated == 0:
                cur.execute(
                    """
                    INSERT INTO overrides (bibcode, scix_id, override)
                    VALUES (%s, %s, %s)
                    """,
                    (bibcode, scix_id, collection),
                )

        self.conn.commit()


class ADSClient:
    def __init__(self):
        self.base_url = ADS_API_URL

    @staticmethod
    def _chunk(items, size):
        for idx in range(0, len(items), size):
            yield items[idx : idx + size]

    def fetch_titles(self, bibcodes, token):
        if not token:
            return {}

        unique_bibcodes = [b for b in dict.fromkeys(bibcodes) if b]
        if not unique_bibcodes:
            return {}

        titles_by_bibcode = {}
        headers = {"Authorization": f"Bearer {token.strip()}"}

        for chunk in self._chunk(unique_bibcodes, 100):
            query = " OR ".join(f'"{bibcode}"' for bibcode in chunk)
            params = {"q": f"bibcode:({query})", "fl": "bibcode,title", "rows": len(chunk)}
            response = requests.get(self.base_url, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            docs = response.json().get("response", {}).get("docs", [])
            for doc in docs:
                bibcode = doc.get("bibcode")
                title = doc.get("title")
                if isinstance(title, list):
                    title = title[0] if title else ""
                if bibcode and title:
                    titles_by_bibcode[bibcode] = title

        return titles_by_bibcode

    def fetch_abstract(self, bibcode, token):
        if not token or not bibcode:
            return ""
        headers = {"Authorization": f"Bearer {token.strip()}"}
        params = {"q": f'bibcode:"{bibcode}"', "fl": "bibcode,abstract", "rows": 1}
        response = requests.get(self.base_url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        docs = response.json().get("response", {}).get("docs", [])
        return docs[0].get("abstract") if docs else ""


class WebAppState:
    def __init__(self):
        self.client = DatabaseClient()
        self.ads = ADSClient()
        self.current_rows: List[Dict[str, Any]] = []
        self.abstract_cache: Dict[str, str] = {}
        self.connected = False


def summarize_exception(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        if response is not None:
            code = response.status_code
            reason = response.reason or ""
            if code == 401:
                return "ADS API unauthorized (401). Check your ADS token."
            return f"ADS API request failed ({code} {reason})."
        return "ADS API request failed."
    first_line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    return first_line[:240]


def extract_scores_map(raw_scores):
    if not raw_scores:
        return {}

    score_obj = None
    try:
        score_obj = json.loads(raw_scores)
    except Exception:
        try:
            score_obj = ast.literal_eval(raw_scores)
        except Exception:
            try:
                score_obj = yaml.safe_load(raw_scores)
            except Exception:
                return {}

    if isinstance(score_obj, dict) and isinstance(score_obj.get("scores"), dict):
        return score_obj["scores"]
    return {}


def format_scores(scores_map):
    if not scores_map:
        return "(No category scores found.)"
    return "\n".join(
        f"{name}: {float(value):.2f}"
        for name, value in sorted(scores_map.items(), key=lambda kv: float(kv[1]), reverse=True)
    )


def json_response(start_response, status_code: int, payload: Dict[str, Any]):
    status = f"{status_code} {'OK' if status_code < 400 else 'ERROR'}"
    body = json.dumps(payload).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response(status, headers)
    return [body]


def html_response(start_response, html: str):
    body = html.encode("utf-8")
    headers = [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response("200 OK", headers)
    return [body]


def read_json_body(environ):
    try:
        length = int(environ.get("CONTENT_LENGTH") or "0")
    except ValueError:
        length = 0
    if length <= 0:
        return {}
    raw = environ["wsgi.input"].read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def serialize_row(row: Dict[str, Any], score_category: str, record_idx: int):
    scores_map = extract_scores_map(row.get("scores"))
    score_val = scores_map.get(score_category)
    try:
        score_display = "" if score_val is None else f"{float(score_val):.2f}"
    except (TypeError, ValueError):
        score_display = ""

    return {
        "record_idx": record_idx,
        "bibcode": row.get("bibcode") or "",
        "title": row.get("title") or "",
        "score": score_display,
        "run_id": row.get("run_id"),
        "validated": bool(row.get("validated")),
        "collection": row.get("collection") or [],
    }


def build_record_detail(row: Dict[str, Any], ads_token: str, state: WebAppState):
    scores_map = extract_scores_map(row.get("scores"))
    collection = row.get("collection") or []
    override = row.get("override") or []

    category_info = []
    collection_set = set(collection)
    override_set = set(override)
    for category in ALLOWED_CATEGORIES:
        raw_score = scores_map.get(category, 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        tags = []
        if category in collection_set:
            tags.append("C")
        if category in override_set:
            tags.append("O")
        category_info.append(
            {
                "name": category,
                "score": f"{score:.2f}",
                "checked": category in collection_set,
                "tags": tags,
            }
        )

    bibcode = row.get("bibcode")
    abstract = state.abstract_cache.get(bibcode)
    if abstract is None:
        abstract = row.get("abstract") or ""
        if not abstract and ads_token and bibcode:
            try:
                abstract = state.ads.fetch_abstract(bibcode=bibcode, token=ads_token)
            except Exception as exc:
                abstract = f"(ADS abstract lookup failed: {summarize_exception(exc)})"
        if not abstract:
            abstract = "(No abstract returned from ADS for this bibcode.)"
        if bibcode:
            state.abstract_cache[bibcode] = abstract

    return {
        "collection_label": f"Current collection: {collection}    Latest override: {override}",
        "scores_text": format_scores(scores_map),
        "abstract_text": abstract,
        "categories": category_info,
    }


def create_app():
    state = WebAppState()
    templates_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )

    def app(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET").upper()

        if path == "/" and method == "GET":
            template = env.get_template("classifier_db_web.html")
            html = template.render(
                allowed_categories=ALLOWED_CATEGORIES,
                query_specs=[spec.label for spec in QUERY_SPECS],
                defaults={
                    "host": os.getenv("PGHOST", "localhost"),
                    "port": os.getenv("PGPORT", "5432"),
                    "dbname": os.getenv("PGDATABASE", "classifier_pipeline"),
                    "user": os.getenv("PGUSER", ""),
                    "password": os.getenv("PGPASSWORD", ""),
                    "ads_token": os.getenv("ADS_API_TOKEN", os.getenv("API_TOKEN", "")),
                    "limit": "200",
                    "score_category": ALLOWED_CATEGORIES[0],
                },
            )
            return html_response(start_response, html)

        if path == "/api/connect" and method == "POST":
            data = read_json_body(environ)
            try:
                state.client.connect(
                    host=str(data.get("host", "")).strip(),
                    port=str(data.get("port", "")).strip(),
                    dbname=str(data.get("dbname", "")).strip(),
                    user=str(data.get("user", "")).strip(),
                    password=str(data.get("password", "")),
                )
                state.connected = True
                return json_response(start_response, 200, {"ok": True, "message": "Connected"})
            except Exception as exc:
                state.connected = False
                return json_response(
                    start_response,
                    400,
                    {"ok": False, "error": summarize_exception(exc), "details": str(exc)},
                )

        if path == "/api/query" and method == "POST":
            if not state.connected or not state.client.conn:
                return json_response(start_response, 400, {"ok": False, "error": "Connect to PostgreSQL first."})

            data = read_json_body(environ)
            preset = str(data.get("preset", "Latest records"))
            run_id = str(data.get("run_id", ""))
            bibcode_term = str(data.get("bibcode_term", ""))
            score_category = str(data.get("score_category", ALLOWED_CATEGORIES[0]))
            ads_token = str(data.get("ads_token", "")).strip()

            try:
                limit = int(data.get("limit", 200))
                if limit <= 0:
                    raise ValueError
            except Exception:
                return json_response(start_response, 400, {"ok": False, "error": "Limit must be a positive integer."})

            spec = QUERY_SPEC_BY_LABEL.get(preset, QUERY_SPECS[0])

            try:
                rows = state.client.run_query(spec=spec, run_id=run_id, bibcode_term=bibcode_term, limit=limit)
                warning = None

                bibcodes = [row.get("bibcode") for row in rows if row.get("bibcode")]
                if bibcodes and ads_token:
                    try:
                        titles = state.ads.fetch_titles(bibcodes, ads_token)
                        for row in rows:
                            bib = row.get("bibcode")
                            if bib in titles:
                                row["title"] = titles[bib]
                    except Exception as exc:
                        warning = summarize_exception(exc)

                state.current_rows = rows
                table_rows = [serialize_row(row, score_category, idx) for idx, row in enumerate(rows)]
                return json_response(
                    start_response,
                    200,
                    {
                        "ok": True,
                        "rows": table_rows,
                        "count": len(rows),
                        "warning": warning,
                    },
                )
            except Exception as exc:
                return json_response(
                    start_response,
                    400,
                    {"ok": False, "error": summarize_exception(exc), "details": str(exc)},
                )

        if path == "/api/record" and method == "GET":
            params = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)
            idx_raw = (params.get("idx") or [""])[0]
            ads_token = (params.get("ads_token") or [""])[0].strip()

            try:
                idx = int(idx_raw)
            except Exception:
                return json_response(start_response, 400, {"ok": False, "error": "Invalid row index."})

            if idx < 0 or idx >= len(state.current_rows):
                return json_response(start_response, 404, {"ok": False, "error": "Row not found."})

            detail = build_record_detail(state.current_rows[idx], ads_token, state)
            return json_response(start_response, 200, {"ok": True, "detail": detail})

        if path == "/api/update" and method == "POST":
            if not state.connected or not state.client.conn:
                return json_response(start_response, 400, {"ok": False, "error": "Connect to PostgreSQL first."})

            data = read_json_body(environ)
            try:
                idx = int(data.get("idx"))
            except Exception:
                return json_response(start_response, 400, {"ok": False, "error": "Invalid row index."})

            if idx < 0 or idx >= len(state.current_rows):
                return json_response(start_response, 404, {"ok": False, "error": "Row not found."})

            selected_categories = data.get("selected_categories") or []
            if not isinstance(selected_categories, list):
                return json_response(start_response, 400, {"ok": False, "error": "selected_categories must be a list."})

            invalid = [cat for cat in selected_categories if cat not in ALLOWED_CATEGORIES]
            if invalid:
                return json_response(start_response, 400, {"ok": False, "error": f"Invalid categories: {invalid}"})

            validated = bool(data.get("validated", True))
            row = state.current_rows[idx]

            try:
                state.client.update_collection(
                    final_collection_id=row.get("final_collection_id"),
                    score_id=row.get("score_id"),
                    bibcode=row.get("bibcode"),
                    scix_id=row.get("scix_id"),
                    collection=selected_categories,
                    validated=validated,
                )
                return json_response(
                    start_response,
                    200,
                    {
                        "ok": True,
                        "message": f"Updated collection to: {selected_categories}",
                    },
                )
            except Exception as exc:
                return json_response(
                    start_response,
                    400,
                    {"ok": False, "error": summarize_exception(exc), "details": str(exc)},
                )

        if path == "/api/health" and method == "GET":
            return json_response(start_response, 200, {"ok": True, "connected": state.connected})

        return json_response(start_response, 404, {"ok": False, "error": "Not found"})

    return app


def main():
    host = os.getenv("CLASSIFIER_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("CLASSIFIER_WEB_PORT", "8000"))

    app = create_app()
    with make_server(host, port, app) as server:
        print(f"Classifier DB web UI running at http://{host}:{port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
