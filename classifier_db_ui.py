#!/usr/bin/env python3
"""Desktop UI to inspect and update classifier_db records."""

import json
import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import psycopg2
import requests
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


ADS_API_URL = "https://api.adsabs.harvard.edu/v1/search/query"


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
            metadata_join = f"""
                LEFT JOIN {self.metadata_table} md ON md.bibcode = s.bibcode
            """
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
                ORDER BY created DESC
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
        collection,
        validated,
    ):
        if not self.conn:
            raise RuntimeError("No database connection.")
        with self.conn.cursor() as cur:
            if final_collection_id:
                cur.execute(
                    """
                    UPDATE final_collection
                    SET collection = %s, validated = %s
                    WHERE id = %s
                    """,
                    (collection, validated, final_collection_id),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO final_collection (bibcode, score_id, collection, validated)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (bibcode, score_id, collection, validated),
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
            payload = response.json()
            docs = payload.get("response", {}).get("docs", [])
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
        payload = response.json()
        docs = payload.get("response", {}).get("docs", [])
        if not docs:
            return ""
        return docs[0].get("abstract") or ""


class CategoryDialog(tk.Toplevel):
    def __init__(self, parent, initial_selection):
        super().__init__(parent)
        self.title("Select Collections")
        self.resizable(False, False)
        self.result = None
        self.vars = {}

        frame = ttk.Frame(self, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="Choose one or more collections:").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        for idx, category in enumerate(ALLOWED_CATEGORIES, start=1):
            var = tk.BooleanVar(value=category in initial_selection)
            self.vars[category] = var
            ttk.Checkbutton(frame, text=category, variable=var).grid(
                row=idx, column=0, sticky="w"
            )

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=len(ALLOWED_CATEGORIES) + 1, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(button_frame, text="Apply", command=self._apply).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_frame, text="Cancel", command=self.destroy).grid(row=0, column=1)

        self.transient(parent)
        self.grab_set()
        self.wait_visibility()
        self.focus_set()

    def _apply(self):
        self.result = [name for name, var in self.vars.items() if var.get()]
        self.destroy()


class ClassifierDbUi(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Classifier DB Inspector")
        self.geometry("1300x760")

        self.client = DatabaseClient()
        self.ads_client = ADSClient()
        self.records_by_item = {}
        self.abstract_cache = {}

        self.host_var = tk.StringVar(value=os.getenv("PGHOST", "localhost"))
        self.port_var = tk.StringVar(value=os.getenv("PGPORT", "5432"))
        self.db_var = tk.StringVar(value=os.getenv("PGDATABASE", "classifier_db"))
        self.user_var = tk.StringVar(value=os.getenv("PGUSER", ""))
        self.password_var = tk.StringVar(value=os.getenv("PGPASSWORD", ""))
        self.ads_token_var = tk.StringVar(
            value=os.getenv("ADS_API_TOKEN", os.getenv("API_TOKEN", ""))
        )
        self.status_var = tk.StringVar(value="Not connected")
        self.query_var = tk.StringVar(value=QUERY_SPECS[0].label)
        self.run_id_var = tk.StringVar()
        self.bibcode_term_var = tk.StringVar()
        self.limit_var = tk.StringVar(value="200")
        self.validated_var = tk.BooleanVar(value=True)

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)

        connection_frame = ttk.LabelFrame(self, text="Database Connection", padding=8)
        connection_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        for i in range(14):
            connection_frame.columnconfigure(i, weight=0)
        connection_frame.columnconfigure(13, weight=1)

        ttk.Label(connection_frame, text="Host").grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(connection_frame, textvariable=self.host_var, width=16).grid(row=0, column=1, padx=(0, 10))
        ttk.Label(connection_frame, text="Port").grid(row=0, column=2, padx=(0, 4))
        ttk.Entry(connection_frame, textvariable=self.port_var, width=7).grid(row=0, column=3, padx=(0, 10))
        ttk.Label(connection_frame, text="DB").grid(row=0, column=4, padx=(0, 4))
        ttk.Entry(connection_frame, textvariable=self.db_var, width=16).grid(row=0, column=5, padx=(0, 10))
        ttk.Label(connection_frame, text="User").grid(row=0, column=6, padx=(0, 4))
        ttk.Entry(connection_frame, textvariable=self.user_var, width=16).grid(row=0, column=7, padx=(0, 10))
        ttk.Label(connection_frame, text="Password").grid(row=0, column=8, padx=(0, 4))
        ttk.Entry(connection_frame, textvariable=self.password_var, width=16, show="*").grid(row=0, column=9, padx=(0, 10))
        ttk.Label(connection_frame, text="ADS Token").grid(row=0, column=10, padx=(0, 4))
        ttk.Entry(connection_frame, textvariable=self.ads_token_var, width=24, show="*").grid(row=0, column=11, padx=(0, 10))
        ttk.Button(connection_frame, text="Connect", command=self._connect).grid(row=0, column=12, padx=(0, 8))
        ttk.Label(connection_frame, textvariable=self.status_var).grid(row=0, column=13, sticky="w")

        query_frame = ttk.LabelFrame(self, text="Query", padding=8)
        query_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        query_frame.columnconfigure(10, weight=1)

        ttk.Label(query_frame, text="Preset").grid(row=0, column=0, padx=(0, 4))
        self.query_combo = ttk.Combobox(
            query_frame,
            textvariable=self.query_var,
            values=[spec.label for spec in QUERY_SPECS],
            width=24,
            state="readonly",
        )
        self.query_combo.grid(row=0, column=1, padx=(0, 10))
        ttk.Label(query_frame, text="run_id").grid(row=0, column=2, padx=(0, 4))
        ttk.Entry(query_frame, textvariable=self.run_id_var, width=10).grid(row=0, column=3, padx=(0, 10))
        ttk.Label(query_frame, text="Bibcode contains").grid(row=0, column=4, padx=(0, 4))
        ttk.Entry(query_frame, textvariable=self.bibcode_term_var, width=18).grid(row=0, column=5, padx=(0, 10))
        ttk.Label(query_frame, text="Limit").grid(row=0, column=6, padx=(0, 4))
        ttk.Entry(query_frame, textvariable=self.limit_var, width=7).grid(row=0, column=7, padx=(0, 10))
        ttk.Button(query_frame, text="Run Query", command=self._run_query).grid(row=0, column=8, padx=(0, 10))
        ttk.Label(query_frame, text="Select a row to inspect and update collections.").grid(row=0, column=10, sticky="e")

        list_frame = ttk.LabelFrame(self, text="Records", padding=8)
        list_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        columns = ("bibcode", "title", "run_id", "validated", "collection")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=18)
        self.tree.heading("bibcode", text="Bibcode")
        self.tree.heading("title", text="Title")
        self.tree.heading("run_id", text="run_id")
        self.tree.heading("validated", text="validated")
        self.tree.heading("collection", text="collection")
        self.tree.column("bibcode", width=170, stretch=False)
        self.tree.column("title", width=650, stretch=True)
        self.tree.column("run_id", width=90, stretch=False)
        self.tree.column("validated", width=90, stretch=False)
        self.tree.column("collection", width=260, stretch=True)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_row_selected)

        y_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=y_scroll.set)

        detail_frame = ttk.LabelFrame(self, text="Details and Update", padding=8)
        detail_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(4, 10))
        detail_frame.rowconfigure(4, weight=1)
        detail_frame.columnconfigure(0, weight=1)

        ttk.Label(detail_frame, text="Current collection / override").grid(row=0, column=0, sticky="w")
        self.collection_label = ttk.Label(detail_frame, text="")
        self.collection_label.grid(row=1, column=0, sticky="w", pady=(0, 6))
        ttk.Checkbutton(
            detail_frame,
            text="Set validated=True when updating",
            variable=self.validated_var,
        ).grid(row=1, column=1, sticky="e", padx=(8, 0))

        button_wrap = ttk.Frame(detail_frame)
        button_wrap.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        for idx, category in enumerate(ALLOWED_CATEGORIES):
            ttk.Button(
                button_wrap,
                text=category,
                command=lambda value=category: self._apply_single_category(value),
            ).grid(row=idx // 4, column=idx % 4, sticky="ew", padx=2, pady=2)
            button_wrap.columnconfigure(idx % 4, weight=1)

        ttk.Button(detail_frame, text="Choose Multiple...", command=self._open_multi_select).grid(
            row=3, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Button(detail_frame, text="Clear Collection", command=self._clear_collection).grid(
            row=3, column=1, sticky="e", pady=(0, 8)
        )

        text_frame = ttk.Frame(detail_frame)
        text_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        text_frame.rowconfigure(1, weight=1)
        text_frame.rowconfigure(3, weight=1)
        text_frame.columnconfigure(0, weight=1)

        ttk.Label(text_frame, text="Scores").grid(row=0, column=0, sticky="w")
        self.scores_text = tk.Text(text_frame, height=9, wrap="word")
        self.scores_text.grid(row=1, column=0, sticky="nsew")
        ttk.Label(text_frame, text="Abstract").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.abstract_text = tk.Text(text_frame, height=9, wrap="word")
        self.abstract_text.grid(row=3, column=0, sticky="nsew")

    def _connect(self):
        try:
            self.client.connect(
                host=self.host_var.get().strip(),
                port=self.port_var.get().strip(),
                dbname=self.db_var.get().strip(),
                user=self.user_var.get().strip(),
                password=self.password_var.get(),
            )
            self.status_var.set("Connected")
        except Exception as exc:
            self.status_var.set("Connection failed")
            messagebox.showerror("Connection failed", str(exc))

    def _current_query_spec(self):
        label = self.query_var.get()
        for spec in QUERY_SPECS:
            if spec.label == label:
                return spec
        return QUERY_SPECS[0]

    def _run_query(self):
        if not self.client.conn:
            messagebox.showwarning("Not connected", "Connect to PostgreSQL first.")
            return

        try:
            limit = int(self.limit_var.get().strip())
            if limit <= 0:
                raise ValueError("Limit must be positive.")
        except Exception:
            messagebox.showwarning("Bad input", "Limit must be a positive integer.")
            return

        try:
            rows = self.client.run_query(
                spec=self._current_query_spec(),
                run_id=self.run_id_var.get(),
                bibcode_term=self.bibcode_term_var.get(),
                limit=limit,
            )
        except Exception as exc:
            messagebox.showerror("Query failed", str(exc))
            return

        bibcodes = [row.get("bibcode") for row in rows if row.get("bibcode")]
        if bibcodes:
            if not self.ads_token_var.get().strip():
                messagebox.showwarning(
                    "ADS token required",
                    "Provide ADS API token (ADS Token field or ADS_API_TOKEN/API_TOKEN env var) "
                    "to fetch titles from ADS.",
                )
            try:
                ads_titles = self.ads_client.fetch_titles(
                    bibcodes=bibcodes, token=self.ads_token_var.get()
                )
                for row in rows:
                    bibcode = row.get("bibcode")
                    if bibcode in ads_titles:
                        row["title"] = ads_titles[bibcode]
            except Exception as exc:
                messagebox.showwarning(
                    "ADS title lookup warning",
                    f"Could not fetch titles from ADS API. Showing rows without ADS titles.\n\n{exc}",
                )

        self.records_by_item.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)

        for row in rows:
            collection_text = ", ".join(row["collection"]) if row.get("collection") else ""
            item_id = self.tree.insert(
                "",
                "end",
                values=(
                    row.get("bibcode") or "",
                    row.get("title") or "",
                    row.get("run_id"),
                    bool(row.get("validated")),
                    collection_text,
                ),
            )
            self.records_by_item[item_id] = row

        self.collection_label.config(text=f"Loaded {len(rows)} rows")
        self._set_text(self.scores_text, "")
        self._set_text(self.abstract_text, "")

    def _on_row_selected(self, _event):
        selected = self.tree.selection()
        if not selected:
            return
        row = self.records_by_item[selected[0]]
        scores_text = self._format_scores(row.get("scores"))
        self._set_text(self.scores_text, scores_text)
        bibcode = row.get("bibcode")
        abstract = self.abstract_cache.get(bibcode, "")
        if not abstract:
            try:
                abstract = self.ads_client.fetch_abstract(
                    bibcode=bibcode, token=self.ads_token_var.get()
                )
                if bibcode:
                    self.abstract_cache[bibcode] = abstract
            except Exception as exc:
                abstract = f"(ADS abstract lookup failed: {exc})"
        self._set_text(
            self.abstract_text,
            abstract or "(No abstract returned from ADS for this bibcode.)",
        )

        collection = row.get("collection") or []
        override = row.get("override") or []
        self.collection_label.config(
            text=f"Current collection: {collection}    Latest override: {override}"
        )

    @staticmethod
    def _set_text(widget: tk.Text, content: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    @staticmethod
    def _format_scores(raw_scores):
        if not raw_scores:
            return "(No scores found.)"
        try:
            score_obj = json.loads(raw_scores)
        except Exception:
            return str(raw_scores)

        scores_map = {}
        if isinstance(score_obj, dict):
            if isinstance(score_obj.get("scores"), dict):
                scores_map = score_obj["scores"]
            else:
                # Fallback in case scores are already a category->score mapping.
                scores_map = {
                    key: value
                    for key, value in score_obj.items()
                    if isinstance(value, (int, float))
                }

        if isinstance(scores_map, dict):
            if not scores_map:
                return "(No category scores found.)"
            return "\n".join(
                f"{name}: {value:.6f}"
                for name, value in sorted(scores_map.items(), key=lambda kv: kv[1], reverse=True)
            )
        return "(No category scores found.)"

    def _selected_row(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No row selected", "Select a record first.")
            return None
        return self.records_by_item[selected[0]]

    def _apply_single_category(self, category):
        self._update_selected_collection([category])

    def _clear_collection(self):
        self._update_selected_collection([])

    def _open_multi_select(self):
        row = self._selected_row()
        if row is None:
            return

        initial = row.get("collection") or []
        dialog = CategoryDialog(self, initial)
        self.wait_window(dialog)
        if dialog.result is None:
            return
        self._update_selected_collection(dialog.result)

    def _update_selected_collection(self, collection):
        row = self._selected_row()
        if row is None:
            return

        invalid = [category for category in collection if category not in ALLOWED_CATEGORIES]
        if invalid:
            messagebox.showerror("Invalid category", f"Invalid categories: {invalid}")
            return

        try:
            self.client.update_collection(
                final_collection_id=row.get("final_collection_id"),
                score_id=row.get("score_id"),
                bibcode=row.get("bibcode"),
                collection=collection,
                validated=bool(self.validated_var.get()),
            )
            self._run_query()
            messagebox.showinfo("Success", f"Updated collection to: {collection}")
        except Exception as exc:
            messagebox.showerror("Update failed", str(exc))

    def _on_close(self):
        self.client.close()
        self.destroy()


def main():
    app = ClassifierDbUi()
    app.mainloop()


if __name__ == "__main__":
    main()
