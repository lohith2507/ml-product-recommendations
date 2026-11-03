"""
Shopping Agent orchestrated with LangGraph.

Uses a ReAct-style tool loop (search, deals, compare, cart) on Vertex Gemini.
Cart writes are only *proposed*; the member confirms in the UI (recommenders/cart.py).

The OpenAI-compatible client loop remains as a fallback for tests and Groq.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from recommenders import cart as cart_service
from recommenders import deals

MAX_TOOL_ROUNDS = 5
MAX_PROPOSALS_PER_TURN = 5
PRODUCTS_SHOWN = 5

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Semantic search over the full catalog. Returns the most relevant products with live price, list price, savings, and rating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What the member is looking for, in natural language."},
                    "category": {"type": "string", "description": "Exact category name to filter on."},
                    "max_price": {"type": "number", "description": "Maximum price in USD."},
                    "min_rating": {"type": "number", "description": "Minimum average rating, 1-5."},
                    "limit": {"type": "integer", "description": "Number of results, 1-10. Default 5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_best_deals",
            "description": "Find the best-value products for a need, ranked by relevance, discount off list price, and rating confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What the member wants a deal on. Empty string for the best deals overall."},
                    "category": {"type": "string", "description": "Exact category name to filter on."},
                    "max_price": {"type": "number", "description": "Maximum price in USD."},
                    "min_rating": {"type": "number", "description": "Minimum average rating, 1-5."},
                    "limit": {"type": "integer", "description": "Number of results, 1-10. Default 5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "Side-by-side price, savings, and rating for specific products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {"type": "array", "items": {"type": "integer"}, "description": "2-5 product IDs."},
                },
                "required": ["product_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cart",
            "description": "The member's current cart: items, quantities, subtotal, and savings.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_cart_change",
            "description": (
                "Propose adding, setting the quantity of, or removing a product in the member's cart. "
                "The change is shown to the member as a confirmation button and only happens if they confirm."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer"},
                    "action": {"type": "string", "enum": ["add", "set", "remove"]},
                    "quantity": {"type": "integer", "description": "Units to add, or the new quantity for 'set'. Default 1."},
                    "reason": {"type": "string", "description": "One short sentence on why, shown to the member."},
                },
                "required": ["product_id", "action"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a shopping agent for a warehouse-club e-commerce store. You help members find products, get the best value, and build their cart.

How to work:
- Always use tools for product facts. Never invent products, prices, discounts, or ratings.
- Use find_best_deals when the member cares about price or value; otherwise search_products.
- When the member asks to add, remove, or change something in the cart, call propose_cart_change. If they refer to an item you showed ("the first one", "the cheaper one"), use that product_id.
- Cart changes are NOT applied until the member presses Confirm. Say "I've prepared that for you to confirm", never "I added it".
- Never handle payment or checkout.
- If the request is ambiguous, ask one short clarifying question.

How to answer:
- Plain text, short and friendly. No markdown tables.
- Mention concrete prices and savings in dollars, e.g. "$18.99 (save $4.99, 21% off)".
- Recommend at most 5 products.

Available categories: {categories}

Member profile:
{profile}

Current cart:
{cart}"""


def _to_float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _message_text(content) -> str:
    """Normalize LangChain / Gemini content blocks to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


class ShoppingAgent:
    """Runs one member turn via LangGraph (default) or the legacy client loop."""

    def __init__(self, clients, profile_fn=None, retriever=None, use_langgraph=True):
        """
        Args:
            clients: list of (client, model) pairs with an OpenAI-compatible
                chat.completions.create; used by the legacy loop / availability.
            profile_fn: user_id -> profile text for the system prompt.
            retriever: optional (query, top_k) -> [{product_id, similarity}] override.
            use_langgraph: when True, orchestrate with LangGraph + Vertex Gemini.
        """
        self.clients = clients
        self.profile_fn = profile_fn
        self.retriever = retriever
        self.use_langgraph = use_langgraph and os.environ.get("USE_LANGGRAPH", "1").lower() not in ("0", "false", "no")

    @classmethod
    def from_llm(cls, llm):
        clients = []
        try:
            from recommenders.vertex_client import agent_clients

            clients.extend(agent_clients())
        except Exception as e:
            print(f"⚠️ Vertex clients not loaded: {e}")

        if llm is not None and getattr(llm, "groq_client", None):
            clients.append((llm.groq_client, os.environ.get("AGENT_MODEL", "openai/gpt-oss-120b")))
            clients.append((llm.groq_client, os.environ.get("AGENT_FALLBACK_MODEL", "openai/gpt-oss-20b")))
        return cls(clients, profile_fn=llm.get_user_profile if llm is not None else None, use_langgraph=True)

    @property
    def available(self):
        return bool(self.clients) or self.use_langgraph

    # ---------- Tools (shared) ----------

    def _run_tool(self, name, args, user_id, turn):
        if name in ("search_products", "find_best_deals"):
            fn = deals.find_best_deals if name == "find_best_deals" else deals.search_products
            max_price = _to_float(args.get("max_price"))
            category = args.get("category") or None
            products = fn(
                str(args.get("query") or ""),
                category=category,
                max_price=max_price,
                min_rating=_to_float(args.get("min_rating")),
                limit=max(1, min(_to_int(args.get("limit"), 5), 10)),
                retriever=self.retriever,
            )
            turn["shown"] = products
            if max_price is not None:
                turn["context"].add(f"Under ${max_price:g}")
            if category:
                turn["context"].add(category)
            if name == "find_best_deals":
                turn["context"].add("Best value")
            return {"products": [deals.compact(p) for p in products]}

        if name == "compare_products":
            ids = args.get("product_ids") or []
            products = deals.get_products([_to_int(i, -1) for i in ids][:5])
            turn["shown"] = products
            return {"products": [deals.compact(p) for p in products]}

        if name == "get_cart":
            cart = cart_service.get_cart(user_id)
            return {
                "items": [{"product_id": i["product_id"], "title": i["title"][:90], "quantity": i["quantity"], "price": i["price"]} for i in cart["items"]],
                "subtotal": cart["subtotal"],
                "savings": cart["savings"],
            }

        if name == "propose_cart_change":
            if len(turn["pending"]) >= MAX_PROPOSALS_PER_TURN:
                return {"error": f"At most {MAX_PROPOSALS_PER_TURN} cart changes per turn."}
            action = cart_service.propose_action(
                user_id,
                _to_int(args.get("product_id"), -1),
                str(args.get("action") or ""),
                _to_int(args.get("quantity"), 1),
                str(args.get("reason") or ""),
            )
            turn["pending"].append(action)
            return {
                "status": "awaiting_member_confirmation",
                "action_id": action["action_id"],
                "product": action["product"]["title"][:90],
            }

        return {"error": f"Unknown tool '{name}'"}

    def _system_prompt(self, user_id):
        from database.models import get_db

        with get_db() as conn:
            categories = [r["category"] for r in conn.execute("SELECT DISTINCT category FROM products ORDER BY 1")]
        try:
            cart = cart_service.get_cart(user_id)
            cart_text = json.dumps(
                {"items": [{"product_id": i["product_id"], "title": i["title"][:60], "quantity": i["quantity"]} for i in cart["items"]], "subtotal": cart["subtotal"]}
            )
        except cart_service.CartError:
            cart_text = "unavailable"
        profile = self.profile_fn(user_id) if self.profile_fn else "unknown"
        return SYSTEM_PROMPT.format(categories=", ".join(categories), profile=profile, cart=cart_text)

    def _fallback(self, message, note):
        products = deals.find_best_deals(message, limit=PRODUCTS_SHOWN, retriever=self.retriever)
        return {
            "response": note,
            "products": products,
            "active_context": ["Best value"],
            "pending_actions": [],
            "tools_used": ["find_best_deals"],
            "orchestrator": "fallback",
        }

    def _finalize(self, answer, turn):
        shown = list(turn["shown"][:PRODUCTS_SHOWN])
        shown_ids = {p["product_id"] for p in shown}
        for action in turn["pending"]:
            if action["product"]["product_id"] not in shown_ids:
                extra = deals.get_products([action["product"]["product_id"]])
                shown.extend(extra)
                shown_ids.update(p["product_id"] for p in extra)

        return {
            "response": answer,
            "products": [{**p, "algorithm": "shopping_agent"} for p in shown],
            "active_context": sorted(turn["context"]),
            "pending_actions": turn["pending"],
            "tools_used": turn["tools"],
        }

    # ---------- LangGraph path ----------

    def _build_langchain_tools(self, user_id, turn):
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        class SearchArgs(BaseModel):
            query: str = Field(..., description="What the member is looking for")
            category: Optional[str] = Field(None, description="Exact category name")
            max_price: Optional[float] = Field(None, description="Maximum price in USD")
            min_rating: Optional[float] = Field(None, description="Minimum average rating 1-5")
            limit: int = Field(5, description="Number of results 1-10")

        class CompareArgs(BaseModel):
            product_ids: list[int] = Field(..., description="2-5 product IDs")

        class CartChangeArgs(BaseModel):
            product_id: int
            action: str = Field(..., description="add | set | remove")
            quantity: int = Field(1, description="Units to add, or new quantity for set")
            reason: str = Field("", description="Short reason shown to the member")

        class EmptyArgs(BaseModel):
            pass

        def _call(name, **kwargs):
            turn["tools"].append(name)
            try:
                result = self._run_tool(name, kwargs, user_id, turn)
            except (cart_service.CartError, ValueError) as e:
                result = {"error": str(e)}
            return json.dumps(result, default=str)

        return [
            StructuredTool.from_function(
                name="search_products",
                description="Semantic search over the full catalog with live price, savings, and rating.",
                func=lambda **kw: _call("search_products", **kw),
                args_schema=SearchArgs,
            ),
            StructuredTool.from_function(
                name="find_best_deals",
                description="Find the best-value products ranked by relevance, discount, and rating.",
                func=lambda **kw: _call("find_best_deals", **kw),
                args_schema=SearchArgs,
            ),
            StructuredTool.from_function(
                name="compare_products",
                description="Side-by-side price, savings, and rating for specific products.",
                func=lambda **kw: _call("compare_products", **kw),
                args_schema=CompareArgs,
            ),
            StructuredTool.from_function(
                name="get_cart",
                description="The member's current cart: items, quantities, subtotal, and savings.",
                func=lambda **kw: _call("get_cart", **kw),
                args_schema=EmptyArgs,
            ),
            StructuredTool.from_function(
                name="propose_cart_change",
                description="Propose a cart add/set/remove. Applied only after the member confirms.",
                func=lambda **kw: _call("propose_cart_change", **kw),
                args_schema=CartChangeArgs,
            ),
        ]

    def _langgraph_model(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from recommenders.vertex_client import chat_location, chat_model, project_id

        return ChatGoogleGenerativeAI(
            model=os.environ.get("VERTEX_CHAT_MODEL") or chat_model(),
            project=project_id(),
            location=chat_location(),
            vertexai=True,
            temperature=0.3,
        )

    def _chat_langgraph(self, user_id, message, conversation_history=None):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langgraph.prebuilt import create_react_agent

        turn = {"shown": [], "pending": [], "context": set(), "tools": []}
        tools = self._build_langchain_tools(user_id, turn)

        try:
            model = self._langgraph_model()
            graph = create_react_agent(model, tools)
        except Exception as e:
            print(f"⚠️ LangGraph init failed: {e}")
            if self.clients:
                return self._chat_legacy(user_id, message, conversation_history)
            return self._fallback(message, "I'm having trouble starting the shopping agent, so here are the best-value matches.")

        messages = [SystemMessage(content=self._system_prompt(user_id))]
        for msg in (conversation_history or [])[-6:]:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=message))

        try:
            result = graph.invoke({"messages": messages}, config={"recursion_limit": MAX_TOOL_ROUNDS * 2 + 2})
        except Exception as e:
            print(f"⚠️ LangGraph invoke failed: {e}")
            if turn["shown"] or turn["pending"]:
                out = self._finalize("Here's what I found.", turn)
                out["orchestrator"] = "langgraph"
                return out
            if self.clients:
                return self._chat_legacy(user_id, message, conversation_history)
            return self._fallback(message, "I'm having trouble reaching the AI agent, so here are the best-value matches for your search.")

        answer = ""
        for msg in reversed(result.get("messages") or []):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                answer = _message_text(msg.content)
                if answer:
                    break
        if not answer:
            answer = "Here's what I found." if (turn["shown"] or turn["pending"]) else ""
        if not answer:
            return self._fallback(message, "I'm having trouble reaching the AI agent, so here are the best-value matches for your search.")

        out = self._finalize(answer, turn)
        out["orchestrator"] = "langgraph"
        return out

    # ---------- Legacy OpenAI-compatible loop (tests / Groq) ----------

    def _complete(self, messages, tool_choice="auto"):
        for client, model in self.clients:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice=tool_choice,
                    temperature=0.3,
                    max_tokens=1200,
                )
                return response.choices[0].message
            except Exception as e:
                print(f"⚠️ Agent model {model} failed: {str(e)[:200]}")
        return None

    def _chat_legacy(self, user_id, message, conversation_history=None):
        if not self.clients:
            return self._fallback(message, "The AI agent isn't configured, so here are the best-value matches for your search.")

        messages = [{"role": "system", "content": self._system_prompt(user_id)}]
        for msg in (conversation_history or [])[-6:]:
            if msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        turn = {"shown": [], "pending": [], "context": set(), "tools": []}
        answer = None

        for round_index in range(MAX_TOOL_ROUNDS + 1):
            reply = self._complete(messages, tool_choice="none" if round_index == MAX_TOOL_ROUNDS else "auto")
            if reply is None:
                break
            if not reply.tool_calls:
                answer = (reply.content or "").strip()
                break

            messages.append({
                "role": "assistant",
                "content": reply.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "thought_signature": getattr(c, "thought_signature", None),
                        "function": {"name": c.function.name, "arguments": c.function.arguments},
                    }
                    for c in reply.tool_calls
                ],
            })
            for call in reply.tool_calls:
                name = call.function.name
                turn["tools"].append(name)
                try:
                    args = json.loads(call.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                    result = self._run_tool(name, args, user_id, turn)
                except (cart_service.CartError, ValueError) as e:
                    result = {"error": str(e)}
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})

        if not answer:
            if turn["shown"] or turn["pending"]:
                answer = "Here's what I found."
            else:
                return self._fallback(message, "I'm having trouble reaching the AI agent, so here are the best-value matches for your search.")

        out = self._finalize(answer, turn)
        out["orchestrator"] = "legacy"
        return out

    def chat(self, user_id, message, conversation_history=None):
        if self.use_langgraph:
            return self._chat_langgraph(user_id, message, conversation_history)
        return self._chat_legacy(user_id, message, conversation_history)
