from __future__ import annotations

import ast
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from simple_solution.utils.data_store import OrderDataStore
from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    ToolCallRecord,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are a grounded Vietnamese electronics order assistant.
Today is {current_day}.

Before calling any tool, extract and verify that the user supplied every required field:
- customer_name
- customer_phone
- customer_email
- shipping_address as one clean string
- items, each with product name and quantity

If any required field is missing, ask one concise Vietnamese clarification question and do not call tools.
Refuse without calling tools if the user asks for fake invoices, manual discount overrides, bypassing stock,
ignoring the catalog, or ignoring policy.

For valid orders, use exactly this grounded workflow:
1. Call list_products to resolve each requested product name. Use enough query text and limit up to 20 when needed.
2. Call get_product_details with exact product_ids returned by list_products. Never invent product IDs.
3. If stock is insufficient, stop and explain; do not price or save.
4. Call get_discount using customer_email as the seed_hint and customer_tier="standard" unless VIP is explicit.
5. Call calculate_order_totals with exact product_id/quantity lines, the detail_token, and discount_rate from get_discount.
6. If pricing is ok, call save_order using these exact top-level fields whenever possible:
   customer_name, customer_phone, customer_email, shipping_address, items, detail_token,
   discount_rate, campaign_code, customer_tier, notes.

Never drop discount_rate or campaign_code after get_discount. Never copy prices, totals, campaign_code,
detail_token, save path, or stock from memory; only use tool outputs.
Final answer must be short Vietnamese and mention order_id, discount/campaign, final_total, and saved path after save.
""".strip()


class DiscountInput(BaseModel):
    customer_email: str = Field(..., description="Customer email. Use this as the stable discount seed.")
    customer_phone: str = Field(default="", description="Fallback seed if email is unavailable.")
    customer_tier: str = Field(default="standard", description="Use standard unless the user explicitly says VIP.")


class SaveOrderToolInput(BaseModel):
    customer_name: str = Field(default="", description="Customer full name. Required before saving.")
    customer_phone: str = Field(default="", description="Customer phone number. Required before saving.")
    customer_email: str = Field(default="", description="Customer email address. Required before saving.")
    shipping_address: str | dict[str, Any] = Field(default="", description="Shipping destination as a clean string.")
    items: list[OrderLineInput] = Field(default_factory=list, description="Exact product IDs and quantities.")
    detail_token: str = Field(default="", description="detail_token returned by get_product_details.")
    discount_rate: float = Field(default=0.0, description="discount_rate returned by get_discount.")
    campaign_code: str = Field(default="", description="campaign_code returned by get_discount.")
    customer_tier: str = Field(default="standard", description="Customer segment returned/used by get_discount.")
    notes: str = Field(default="", description="Optional internal note.")
    customer: dict[str, Any] = Field(default_factory=dict, description="Fallback nested customer object if needed.")
    address: dict[str, Any] | str = Field(default="", description="Fallback nested address if needed.")
    discount: dict[str, Any] = Field(default_factory=dict, description="Fallback nested discount object if needed.")


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 20,
    ) -> str:
        """Search the local catalog by product name/brand/features and return candidate product IDs; call before details."""
        payload = store.list_products(
            query=(query or "").strip() or None,
            category=(category or "").strip() or None,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact price, stock, SKU, and detail_token for product IDs returned by list_products."""
        product_ids = _coerce_product_ids(product_ids)
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(customer_email: str, customer_phone: str = "", customer_tier: str = "standard") -> str:
        """Return discount_rate and campaign_code; pass both unchanged into calculate_order_totals/save_order."""
        seed_hint = (customer_email or "").strip() or (customer_phone or "").strip() or "guest"
        normalized_tier = "vip" if str(customer_tier).strip().lower() == "vip" else "standard"
        return json.dumps(store.get_discount(seed_hint=seed_hint, customer_tier=normalized_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[OrderLineInput], detail_token: str, discount_rate: float) -> str:
        """Validate detail_token and stock, then calculate subtotal, discount, and final_total for exact product IDs."""
        items = _coerce_items(items)
        payload = store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderToolInput)
    def save_order(
        customer_name: str = "",
        customer_phone: str = "",
        customer_email: str = "",
        shipping_address: str | dict[str, Any] = "",
        items: list[OrderLineInput] | None = None,
        detail_token: str = "",
        discount_rate: float = 0.0,
        campaign_code: str = "",
        customer_tier: str = "standard",
        notes: str = "",
        customer: dict[str, Any] | None = None,
        address: dict[str, Any] | str = "",
        discount: dict[str, Any] | None = None,
    ) -> str:
        """Persist the confirmed order; requires complete customer fields, exact items, detail_token, discount_rate, and campaign_code."""
        payload = _normalize_order_payload(
            {
                "customer_name": customer_name,
                "customer_phone": customer_phone,
                "customer_email": customer_email,
                "shipping_address": shipping_address or address,
                "items": items or [],
                "detail_token": detail_token,
                "discount_rate": discount_rate,
                "campaign_code": campaign_code,
                "customer_tier": customer_tier,
                "notes": notes,
                "customer": customer or {},
                "address": address,
                "discount": discount or {},
            }
        )
        validation_errors = _validate_order_payload(payload)
        if validation_errors:
            return json.dumps({"status": "error", "errors": validation_errors}, ensure_ascii=False)
        result = store.save_order(
            customer_name=payload["customer_name"],
            customer_phone=payload["customer_phone"],
            customer_email=payload["customer_email"],
            shipping_address=payload["shipping_address"],
            items=payload["items"],
            detail_token=payload["detail_token"],
            discount_rate=payload["discount_rate"],
            campaign_code=payload["campaign_code"],
            customer_tier=payload["customer_tier"],
            notes=payload["notes"],
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    preflight_answer = build_preflight_response(query)
    if preflight_answer:
        return AgentResult(
            query=query,
            final_answer=preflight_answer,
            tool_calls=[],
            provider=provider,
            model_name=model_name,
            saved_order=None,
            saved_order_path=None,
        )

    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def build_preflight_response(query: str) -> str:
    normalized = _normalize_text(query)
    if _looks_like_guardrail_violation(normalized):
        return (
            "Mình không thể tạo hóa đơn giả, bỏ qua catalog/tồn kho, hoặc tự ép khuyến mãi. "
            "Mình chỉ có thể tạo đơn theo sản phẩm thật, tồn kho thật và khuyến mãi từ hệ thống."
        )

    missing_fields = _missing_required_order_fields(query)
    if missing_fields:
        return "Mình cần thêm " + ", ".join(missing_fields) + " trước khi tạo đơn hàng."
    return ""


def _looks_like_guardrail_violation(normalized_query: str) -> bool:
    guardrail_terms = [
        "hoa don gia",
        "fake invoice",
        "giam gia 90",
        "ep giam gia",
        "tu ep giam gia",
        "manual discount",
        "bo qua ton kho",
        "bypass stock",
        "ignore stock",
        "bo qua catalog",
        "khong can theo catalog",
        "ignore catalog",
        "bo qua policy",
        "ignore policy",
    ]
    return any(term in normalized_query for term in guardrail_terms)


def _missing_required_order_fields(query: str) -> list[str]:
    missing: list[str] = []
    normalized = _normalize_text(query)
    if not _extract_name(query):
        missing.append("tên khách hàng")
    if not re.search(r"(?<!\d)0\d{8,10}(?!\d)", query):
        missing.append("số điện thoại")
    if not re.search(r"[\w.+-]+@[\w.-]+\.\w+", query):
        missing.append("email")
    address_markers = [
        "giao den",
        "giao toi",
        "giao hang den",
        "dia chi giao hang",
        "ship to",
        "giao ve",
    ]
    if not any(marker in normalized for marker in address_markers):
        missing.append("địa chỉ giao hàng")
    if not _extract_item_mentions(query):
        missing.append("sản phẩm và số lượng")
    return missing


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return " ".join(stripped.lower().split())


def _extract_name(query: str) -> str:
    patterns = [
        r"(?:tạo|tao|lưu|luu|create)[^.!?\n]{0,40}?\bcho\s+(.+?)(?:,|\.|\bsố điện thoại\b|\bso dien thoai\b|\bemail\b|\bphone\b|\bgiao\b|\bship\b)",
        r"\bcho\s+(.+?)(?:,|\.|\bsố điện thoại\b|\bso dien thoai\b|\bemail\b|\bphone\b|\bgiao\b|\bship\b)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = re.sub(r"^(anh|chị|chi|bạn|ban|customer)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
        normalized = _normalize_text(candidate)
        if candidate and "cong ty" not in normalized and "company" not in normalized:
            return candidate
    return ""


def _extract_item_mentions(query: str) -> list[str]:
    mentions = re.findall(r"(?<!\d)(\d+)\s+([^,.;\n]+)", query)
    product_mentions = []
    ignored_terms = ("số điện thoại", "so dien thoai", "phone", "email", "quận", "quan", "tầng", "tang")
    for quantity, text in mentions:
        normalized = _normalize_text(text)
        if int(quantity) > 0 and not any(term in normalized for term in ignored_terms):
            product_mentions.append(text.strip())
    return product_mentions


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            continue
        return payload.get("saved_order"), payload.get("path")
    return None, None


def _normalize_order_payload(raw: Any) -> dict[str, Any]:
    payload = _coerce_object(raw)
    customer = _coerce_object(payload.get("customer", {}))
    discount = _coerce_object(payload.get("discount", {}))

    normalized = {
        "customer_name": _first_text(payload, "customer_name", "name") or _first_text(customer, "name", "customer_name"),
        "customer_phone": _first_text(payload, "customer_phone", "phone") or _first_text(customer, "phone", "customer_phone"),
        "customer_email": _first_text(payload, "customer_email", "email") or _first_text(customer, "email", "customer_email"),
        "shipping_address": _normalize_address(
            payload.get("shipping_address")
            or payload.get("address")
            or payload.get("ship_to")
            or customer.get("shipping_address")
            or customer.get("address")
        ),
        "items": _coerce_items(payload.get("items", [])),
        "detail_token": str(payload.get("detail_token", "")).strip(),
        "discount_rate": _coerce_discount_rate(payload.get("discount_rate"), discount.get("discount_rate")),
        "campaign_code": _first_text(payload, "campaign_code") or _first_text(discount, "campaign_code"),
        "customer_tier": (_first_text(payload, "customer_tier") or _first_text(discount, "customer_tier") or "standard"),
        "notes": str(payload.get("notes", "") or "").strip(),
    }
    return normalized


def _validate_order_payload(payload: dict[str, Any]) -> list[str]:
    required_text_fields = [
        "customer_name",
        "customer_phone",
        "customer_email",
        "shipping_address",
        "detail_token",
        "campaign_code",
    ]
    errors = [f"Missing required field: {field}." for field in required_text_fields if not payload.get(field)]
    if not payload.get("items"):
        errors.append("Missing required field: items.")
    if payload.get("discount_rate") not in {0.0, 0.1, 0.2}:
        errors.append("discount_rate must come from get_discount and be 0.0, 0.1, or 0.2.")
    return errors


def _first_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalize_address(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, dict):
        preferred_keys = (
            "line1",
            "street",
            "address",
            "ward",
            "district",
            "city",
            "province",
            "state",
            "country",
        )
        parts = [str(raw[key]).strip() for key in preferred_keys if raw.get(key)]
        if not parts:
            parts = [str(value).strip() for value in raw.values() if value]
        return ", ".join(part for part in parts if part)
    if isinstance(raw, list):
        return ", ".join(str(item).strip() for item in raw if str(item).strip())
    return re.sub(r"\s+", " ", str(raw).strip())


def _coerce_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _coerce_discount_rate(primary: Any, fallback: Any) -> float:
    primary_value = _coerce_float(primary)
    fallback_value = _coerce_float(fallback)
    if fallback_value in {0.0, 0.1, 0.2} and fallback is not None:
        return fallback_value
    if primary_value in {0.0, 0.1, 0.2}:
        return primary_value
    return fallback_value


def _coerce_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}
    return {}


def _coerce_product_ids(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in re.split(r"[,\s]+", text) if item.strip()]
    return []


def _coerce_items(raw: Any) -> list[OrderLineInput]:
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        text = raw.strip()
        items = []
        if text:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    continue
                if isinstance(parsed, list):
                    items = parsed
                    break
            if not items:
                for piece in text.split(","):
                    piece = piece.strip()
                    if not piece:
                        continue
                    if ":" in piece:
                        product_id, qty = piece.split(":", 1)
                        items.append({"product_id": product_id.strip(), "quantity": qty.strip()})
    else:
        items = []

    normalized: list[OrderLineInput] = []
    for item in items:
        if isinstance(item, OrderLineInput):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            product_id = _first_text(item, "product_id", "id", "sku", "product")
            try:
                quantity = int(item.get("quantity", item.get("qty", 1)))
            except (TypeError, ValueError):
                continue
            if product_id and quantity > 0:
                normalized.append(OrderLineInput(product_id=product_id, quantity=quantity))
    return normalized
