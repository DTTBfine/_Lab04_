from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.core.schemas import OrderLineInput, ProductRecord


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    compact = re.sub(r"[^a-zA-Z0-9]+", " ", stripped.lower())
    return re.sub(r"\s+", " ", compact).strip()


def _normalize_phone(value: str) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _normalize_email(value: str) -> str:
    return str(value or "").strip().lower()


def _normalize_free_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _campaign_for_rate(discount_rate: float) -> str:
    if discount_rate == 0.0:
        return "NO-PROMO"
    return f"FLASH-{int(discount_rate * 100):02d}"


class OrderDataStore:
    def __init__(self, data_dir: Path, output_dir: Path, *, today: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.today = today or "2026-06-01"
        raw_products = json.loads((self.data_dir / "products.json").read_text(encoding="utf-8"))
        self.products = [ProductRecord(**item) for item in raw_products]
        self.product_index = {item.product_id: item for item in self.products}
        self.product_positions = {item.product_id: index for index, item in enumerate(self.products)}
        self.category_aliases = {
            "laptop": "laptop",
            "notebook": "laptop",
            "monitor": "monitor",
            "screen": "monitor",
            "man hinh": "monitor",
            "mouse": "mouse",
            "chuot": "mouse",
            "keyboard": "keyboard",
            "ban phim": "keyboard",
            "headphone": "headphone",
            "tai nghe": "headphone",
            "dock": "dock",
            "storage": "storage",
            "ssd": "storage",
            "stand": "stand",
            "webcam": "webcam",
        }

    @staticmethod
    def build_detail_token(product_ids: list[str]) -> str:
        normalized = "|".join(sorted(product_ids))
        return "DET-" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10].upper()

    @staticmethod
    def build_order_fingerprint(
        *,
        today: str,
        customer_email: str,
        customer_phone: str,
        shipping_address: str,
        items: list[dict],
        discount_rate: float,
        campaign_code: str,
    ) -> str:
        seed = json.dumps(
            {
                "today": today,
                "customer_email": customer_email,
                "customer_phone": customer_phone,
                "shipping_address": shipping_address,
                "items": items,
                "discount_rate": discount_rate,
                "campaign_code": campaign_code,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return "FP-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12].upper()

    def validate_detail_token(self, product_ids: list[str], detail_token: str) -> bool:
        return detail_token == self.build_detail_token(product_ids)

    def canonicalize_category(self, value: str | None) -> str | None:
        if not value:
            return None
        return self.category_aliases.get(_normalize(value), _normalize(value))

    def list_products(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> list[dict]:
        normalized_query = _normalize(query or "")
        query_terms = [term for term in normalized_query.split() if term and (len(term) > 1 or term.isdigit())]
        wanted_category = self.canonicalize_category(category)
        wanted_tags = {_normalize(tag) for tag in (required_tags or []) if tag.strip()}
        results: list[tuple[int, int, int, int, str, dict]] = []

        for product in self.products:
            if in_stock_only and product.stock <= 0:
                continue
            if wanted_category and product.category != wanted_category:
                continue
            if max_unit_price is not None and product.unit_price > max_unit_price:
                continue

            normalized_name = _normalize(product.name)
            normalized_sku = _normalize(product.sku)
            haystack = _normalize(" ".join([product.name, product.sku, product.brand, product.category, product.description, *product.tags]))
            score = 0
            matched_terms: list[str] = []
            exact_rank = 0
            if normalized_query:
                if normalized_query == normalized_name:
                    score += 100
                    exact_rank = 3
                elif normalized_query in {normalized_sku, _normalize(product.product_id)}:
                    score += 100
                    exact_rank = 3
                elif normalized_query in normalized_name:
                    score += 40
                    exact_rank = 2
                elif normalized_name in normalized_query:
                    score += 30
                    exact_rank = 1
            for term in query_terms:
                if term == normalized_name or term == normalized_sku:
                    score += 20
                    matched_terms.append(term)
                elif re.search(rf"\b{re.escape(term)}\b", haystack):
                    score += 4 if term.isdigit() else 3
                    matched_terms.append(term)
                elif term in haystack:
                    score += 1
                    matched_terms.append(term)
            for tag in wanted_tags:
                if tag in haystack:
                    score += 5
                    matched_terms.append(tag)
                else:
                    score -= 2
            if wanted_category:
                score += 6
            if query_terms and not matched_terms:
                continue
            results.append(
                (
                    score,
                    exact_rank,
                    len(set(matched_terms)),
                    product.stock,
                    product.product_id,
                    {
                        "product_id": product.product_id,
                        "name": product.name,
                        "brand": product.brand,
                        "category": product.category,
                        "tags": product.tags,
                        "matched_terms": sorted(set(matched_terms)),
                        "debug": {
                            "score": score,
                            "exact_rank": exact_rank,
                        },
                        "next_step": "Call get_product_details with the chosen product_id list to verify price, stock, and the detail_token.",
                    },
                )
            )

        results.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], self.product_positions[item[4]], item[4]))
        return [item[-1] for item in results[:limit]]

    def get_product_details(self, product_ids: list[str]) -> dict:
        details: list[dict] = []
        for product_id in product_ids:
            product = self.product_index.get(product_id)
            if not product:
                details.append({"product_id": product_id, "status": "not_found"})
                continue
            details.append(
                {
                    "status": "ok",
                    "product_id": product.product_id,
                    "sku": product.sku,
                    "name": product.name,
                    "brand": product.brand,
                    "category": product.category,
                    "unit_price": product.unit_price,
                    "stock": product.stock,
                    "warranty_months": product.warranty_months,
                    "tags": product.tags,
                    "description": product.description,
                }
            )
        found_product_ids = [item["product_id"] for item in details if item.get("status") == "ok"]
        missing_product_ids = [item["product_id"] for item in details if item.get("status") != "ok"]
        return {
            "status": "ok" if found_product_ids and not missing_product_ids else "error",
            "detail_token": self.build_detail_token(found_product_ids) if found_product_ids else "",
            "items": details,
            "errors": [f"Unknown product_id: {product_id}." for product_id in missing_product_ids],
        }

    def get_discount(self, *, seed_hint: str, customer_tier: str = "standard") -> dict:
        normalized_seed = seed_hint.strip().lower()
        normalized_tier = customer_tier.strip().lower() or "standard"
        if normalized_tier in {"none", "no_promo", "no-promo", "no promotion", "no_discount", "no-discount"}:
            discount_rate = 0.0
            normalized_tier = "standard"
        else:
            digest = hashlib.sha256(f"{normalized_tier}|{normalized_seed}".encode("utf-8")).hexdigest()
            discount_rate = 0.2 if int(digest[-2:], 16) % 10 < 4 else 0.1
        return {
            "status": "ok",
            "seed_hint": seed_hint,
            "customer_tier": normalized_tier,
            "discount_rate": discount_rate,
            "campaign_code": _campaign_for_rate(discount_rate),
        }

    def calculate_order_totals(self, *, items: list[OrderLineInput], detail_token: str, discount_rate: float) -> dict:
        if discount_rate not in {0.0, 0.1, 0.2}:
            return {"status": "error", "errors": [f"Unsupported discount rate: {discount_rate}."]}
        normalized_items = self.normalize_items(items)
        requested_product_ids = [item.product_id for item in normalized_items]
        if not self.validate_detail_token(requested_product_ids, detail_token):
            return {
                "status": "error",
                "errors": ["Invalid detail token. Call get_product_details again before pricing this order."],
                "debug": {
                    "expected_detail_token": self.build_detail_token(requested_product_ids),
                    "received_detail_token": detail_token,
                },
            }

        errors: list[str] = []
        lines: list[dict] = []
        subtotal = 0
        for item in sorted(normalized_items, key=lambda current: current.product_id):
            product = self.product_index.get(item.product_id)
            if not product:
                errors.append(f"Unknown product_id: {item.product_id}.")
                continue
            if item.quantity <= 0:
                errors.append(f"Invalid quantity for {product.name}: {item.quantity}.")
                continue
            if item.quantity > product.stock:
                errors.append(
                    f"Insufficient stock for {product.name}: requested {item.quantity}, available {product.stock}."
                )
                continue
            line_total = product.unit_price * item.quantity
            subtotal += line_total
            lines.append(
                {
                    "product_id": product.product_id,
                    "sku": product.sku,
                    "name": product.name,
                    "category": product.category,
                    "quantity": item.quantity,
                    "unit_price": product.unit_price,
                    "line_total": line_total,
                }
            )

        if errors:
            return {"status": "error", "errors": errors, "items": lines}

        discount_amount = int(subtotal * discount_rate)
        final_total = subtotal - discount_amount
        return {
            "status": "ok",
            "items": lines,
            "pricing": {
                "currency": "VND",
                "subtotal": subtotal,
                "discount_rate": discount_rate,
                "discount_amount": discount_amount,
                "final_total": final_total,
            },
            "detail_token": detail_token,
        }

    def validate_campaign(self, *, discount_rate: float, campaign_code: str) -> list[str]:
        if discount_rate not in {0.0, 0.1, 0.2}:
            return [f"Unsupported discount rate: {discount_rate}."]
        expected_campaign = _campaign_for_rate(discount_rate)
        if campaign_code != expected_campaign:
            return [f"campaign_code must be {expected_campaign} for discount_rate {discount_rate}, got {campaign_code!r}."]
        return []

    def normalize_items(self, items: list[OrderLineInput]) -> list[OrderLineInput]:
        merged: dict[str, int] = {}
        order: list[str] = []
        for item in items:
            if item.product_id not in merged:
                order.append(item.product_id)
                merged[item.product_id] = 0
            merged[item.product_id] += int(item.quantity)
        return [OrderLineInput(product_id=product_id, quantity=merged[product_id]) for product_id in order]

    def validate_customer_fields(
        self,
        *,
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
    ) -> list[str]:
        errors: list[str] = []
        if not customer_name:
            errors.append("Missing required field: customer_name.")
        if not customer_phone:
            errors.append("Missing required field: customer_phone.")
        elif not re.fullmatch(r"0\d{8,10}", customer_phone):
            errors.append(f"Invalid customer_phone: {customer_phone!r}.")
        if not customer_email:
            errors.append("Missing required field: customer_email.")
        elif not re.fullmatch(r"[\w.+-]+@[\w.-]+\.\w+", customer_email):
            errors.append(f"Invalid customer_email: {customer_email!r}.")
        if not shipping_address:
            errors.append("Missing required field: shipping_address.")
        return errors

    def validate_saved_payload(self, payload: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("order_id", "created_at", "status", "customer", "items", "pricing", "discount", "save_path", "source"):
            if key not in payload:
                errors.append(f"Saved payload missing key: {key}.")
        customer = payload.get("customer", {})
        if isinstance(customer, dict):
            errors.extend(
                self.validate_customer_fields(
                    customer_name=str(customer.get("name", "")),
                    customer_phone=str(customer.get("phone", "")),
                    customer_email=str(customer.get("email", "")),
                    shipping_address=str(customer.get("shipping_address", "")),
                )
            )
        else:
            errors.append("Saved payload customer must be an object.")
        pricing = payload.get("pricing", {})
        discount = payload.get("discount", {})
        if isinstance(pricing, dict) and isinstance(discount, dict):
            try:
                discount_rate = float(pricing.get("discount_rate", -1))
            except (TypeError, ValueError):
                errors.append(f"Invalid pricing.discount_rate: {pricing.get('discount_rate')!r}.")
                discount_rate = -1.0
            errors.extend(
                self.validate_campaign(
                    discount_rate=discount_rate,
                    campaign_code=str(discount.get("campaign_code", "")),
                )
            )
        return errors

    def save_order(
        self,
        *,
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> dict:
        normalized_customer_name = _normalize_free_text(customer_name)
        normalized_customer_phone = _normalize_phone(customer_phone)
        normalized_customer_email = _normalize_email(customer_email)
        normalized_shipping_address = _normalize_free_text(shipping_address)
        normalized_customer_tier = _normalize_free_text(customer_tier).lower() or "standard"
        normalized_campaign_code = _normalize_free_text(campaign_code)
        normalized_items = self.normalize_items(items)

        validation_errors = self.validate_customer_fields(
            customer_name=normalized_customer_name,
            customer_phone=normalized_customer_phone,
            customer_email=normalized_customer_email,
            shipping_address=normalized_shipping_address,
        )
        validation_errors.extend(self.validate_campaign(discount_rate=discount_rate, campaign_code=normalized_campaign_code))
        if validation_errors:
            return {"status": "error", "errors": validation_errors}

        pricing_snapshot = self.calculate_order_totals(
            items=normalized_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        if pricing_snapshot["status"] != "ok":
            return pricing_snapshot

        normalized_item_payload = sorted(
            [{"product_id": item.product_id, "quantity": item.quantity} for item in normalized_items],
            key=lambda current: current["product_id"],
        )
        seed_payload = json.dumps(
            {
                "customer_email": normalized_customer_email,
                "customer_phone": normalized_customer_phone,
                "items": normalized_item_payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        order_id = "ORD-" + hashlib.sha1(seed_payload.encode("utf-8")).hexdigest()[:10].upper()
        relative_path = Path("artifacts") / "orders" / f"{order_id}.json"
        absolute_path = self.output_dir / f"{order_id}.json"

        payload = {
            "order_id": order_id,
            "created_at": self.today,
            "status": "confirmed",
            "customer": {
                "name": normalized_customer_name,
                "phone": normalized_customer_phone,
                "email": normalized_customer_email,
                "shipping_address": normalized_shipping_address,
            },
            "items": pricing_snapshot["items"],
            "pricing": pricing_snapshot["pricing"],
            "discount": {
                "campaign_code": normalized_campaign_code,
                "customer_tier": normalized_customer_tier,
            },
            "notes": _normalize_free_text(notes),
            "save_path": str(relative_path),
            "absolute_save_path": str(absolute_path),
            "order_fingerprint": self.build_order_fingerprint(
                today=self.today,
                customer_email=normalized_customer_email,
                customer_phone=normalized_customer_phone,
                shipping_address=normalized_shipping_address,
                items=normalized_item_payload,
                discount_rate=discount_rate,
                campaign_code=normalized_campaign_code,
            ),
            "source": "llm-order-agent",
        }

        payload_errors = self.validate_saved_payload(payload)
        if payload_errors:
            return {"status": "error", "errors": payload_errors}

        try:
            absolute_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            return {"status": "error", "errors": [f"Failed to write order file: {exc}."], "path": str(absolute_path)}
        return {
            "status": "saved",
            "order_id": order_id,
            "path": str(absolute_path),
            "relative_path": str(relative_path),
            "saved_order": payload,
        }
