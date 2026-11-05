"""
Shopping Agent orchestrated with LangGraph.

Uses a ReAct-style tool loop (search, deals, compare, cart) on Vertex Gemini.
Cart writes are only *proposed*; the member confirms in the UI (recommenders/cart.py).

The OpenAI-compatible client loop remains as a fallback for tests and Groq.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
from typing import Optional

from recommenders import budget_kit
from recommenders import cart as cart_service
from recommenders import chat_memory
from recommenders import checkout as checkout_service
from recommenders import deal_watch
from recommenders import deals
from recommenders import member_prefs
from recommenders import wishlist as wishlist_service
from recommenders.store import get_thread_store

MAX_TOOL_ROUNDS = 5
MAX_PROPOSALS_PER_TURN = 5
PRODUCTS_SHOWN = 5

# Per-request tool context so the compiled LangGraph can be process-cached.
_turn_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("shopping_agent_turn")

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
    {
        "type": "function",
        "function": {
            "name": "recall_past_chats",
            "description": "Search this member's past chat memory (RAG) for similar prior turns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to recall, e.g. backpacks under 40."},
                    "limit": {"type": "integer", "description": "1-5 memories. Default 3."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_wishlist",
            "description": "List products on the member's wishlist.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_wishlist",
            "description": "Add or remove a product on the wishlist immediately (no confirm step).",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer"},
                    "action": {"type": "string", "enum": ["add", "remove"], "description": "Default add."},
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_deal",
            "description": "Explain why a product is a good or weak deal: savings, rating confidence, deal score inputs.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "integer"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cart_savings_summary",
            "description": "Summarize the live cart subtotal and savings versus list price.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_next_questions",
            "description": "Suggest 3 short follow-up questions based on products just shown or past memory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_cheaper_match",
            "description": "Find cheaper substitutes for a product_id or a free-text description (e.g. from an uploaded photo).",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "Anchor product to beat on price."},
                    "query": {"type": "string", "description": "Description if no product_id (image caption ok)."},
                    "limit": {"type": "integer", "description": "1-8 results. Default 5."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_substitute",
            "description": "Explain with a savings receipt why an alternative is better or worse than the original.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "Original product."},
                    "alternative_id": {"type": "integer", "description": "Substitute product."},
                },
                "required": ["product_id", "alternative_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_budget_kit",
            "description": "Build a multi-item kit under a dollar budget (e.g. $100 back-to-school: backpack, lunchbox).",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "description": "Max spend in USD."},
                    "needs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of item needs, one product each.",
                    },
                    "category": {"type": "string", "description": "Optional category filter."},
                },
                "required": ["budget", "needs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "watch_deal",
            "description": "Watch a product for price drops (optionally under a target price). Creates a deal alert when price falls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer"},
                    "target_price": {"type": "number", "description": "Optional max price to alert at."},
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_deal_alerts",
            "description": "List recent price-drop alerts for this member.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_member_prefs",
            "description": "Read structured prefs: preferred/avoid brands, sizes, max budget.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_member_prefs",
            "description": "Update preferred_brands, avoid_brands, sizes map, max_budget, or notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "preferred_brands": {"type": "array", "items": {"type": "string"}},
                    "avoid_brands": {"type": "array", "items": {"type": "string"}},
                    "sizes": {"type": "object", "additionalProperties": {"type": "string"}},
                    "max_budget": {"type": "number"},
                    "notes": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_checkout_links",
            "description": "Build store/Amazon checkout deep links for the current cart (handoff, not payment).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_site",
            "description": (
                "Operate the website for the member. Use this when they ask to open a page, search, filter, "
                "switch recommendation algorithms, open the cart, check out, show a product, summarize reviews, "
                "or add/remove/set cart items. Explicit cart requests are applied immediately."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "description": "Up to 8 actions, in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "navigate", "search", "filter_category", "switch_algo",
                                        "open_cart", "close_cart", "checkout", "show_product",
                                        "summarize_reviews", "add_to_cart", "remove_from_cart", "set_cart_qty",
                                    ],
                                },
                                "section": {"type": "string", "description": "recommendations | products | analytics | abtest | admin"},
                                "query": {"type": "string"},
                                "category": {"type": "string"},
                                "algo": {"type": "string", "description": "collaborative | content | hybrid"},
                                "product_id": {"type": "integer"},
                                "quantity": {"type": "integer"},
                            },
                            "required": ["type"],
                        },
                    }
                },
                "required": ["actions"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a shopping agent for a warehouse-club e-commerce store. You help members find products, get the best value, and build their cart.

How to work:
- Always use tools for product facts. Never invent products, prices, discounts, or ratings.
- Use find_best_deals when the member cares about price or value; otherwise search_products.
- When the member says "again", "like last time", or refers to prior chats, call recall_past_chats first.
- When they ask why a deal is good, call explain_deal. When comparing a substitute, call explain_substitute.
- For "cheaper than this", photo matches, or "find something cheaper", call find_cheaper_match.
- For budget kits ("$100 back to school"), call plan_budget_kit with a needs list.
- For price watches / "notify me when cheaper", call watch_deal. For recent drops, get_deal_alerts.
- Respect member prefs (get_member_prefs / update_member_prefs). Soft-filter avoid brands.
- Wishlist changes use update_wishlist (applied immediately). Cart changes use propose_cart_change and need Confirm.
- When the member tells you to do something on the site (open a section, search, filter, switch Collaborative/Content/AI Hybrid, show a product, summarize reviews, open the cart, or check out), call control_site. That runs on the page.
- When they explicitly say to add, remove, or change a cart quantity, call control_site. That applies immediately. Say what you did.
- Use propose_cart_change only when you are suggesting a cart change they did not ask for. Then say it is ready to confirm.
- Use cart_savings_summary when they ask about cart total or savings.
- For checkout / buy now / amazon link, call get_checkout_links (handoff URLs only; you do not take payment).
- Optionally call suggest_next_questions after showing products.
- Never collect card numbers or process payment yourself.
- If the request is ambiguous, ask one short clarifying question.

How to answer:
- Plain text, short and friendly. No markdown tables.
- Mention concrete prices and savings in dollars, e.g. "$18.99 (save $4.99, 21% off)".
- When you have a receipt (explain_deal / explain_substitute / find_cheaper_match), quote the numbers.
- Recommend at most 5 products.

Available categories: {categories}

Member profile:
{profile}

Structured prefs:
{prefs}

Current cart:
{cart}

Relevant past chats (RAG memory):
{memory}"""


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

    _graph = None
    _graph_lock = threading.Lock()

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
        # New process / rebind: drop cached graph so tool list stays current.
        cls._graph = None
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

        if name == "recall_past_chats":
            memories = chat_memory.retrieve_memory(
                user_id,
                str(args.get("query") or ""),
                top_k=max(1, min(_to_int(args.get("limit"), 3), 5)),
            )
            turn["context"].add("Past chats")
            return {"memories": memories}

        if name == "get_wishlist":
            items = wishlist_service.get_wishlist(user_id)
            return {"items": [deals.compact(p) for p in items], "count": len(items)}

        if name == "update_wishlist":
            items = wishlist_service.update_wishlist(
                user_id,
                _to_int(args.get("product_id"), -1),
                str(args.get("action") or "add"),
            )
            turn["context"].add("Wishlist")
            return {"items": [deals.compact(p) for p in items], "count": len(items)}

        if name == "explain_deal":
            pid = _to_int(args.get("product_id"), -1)
            products = deals.get_products([pid])
            if not products:
                return {"error": f"Unknown product {pid}"}
            p = products[0]
            metrics = deals.deal_metrics(p)
            relevance = float(p.get("relevance") or 0.5)
            score = deals.deal_score(p, relevance)
            turn["shown"] = products
            return {
                "product": deals.compact(p),
                "savings": metrics["savings"],
                "savings_pct": round(metrics["savings_pct"] * 100, 1),
                "deal_score": score,
                "why": (
                    f"${p.get('price')} vs list ${p.get('list_price') or p.get('price')}, "
                    f"save ${metrics['savings']} ({round(metrics['savings_pct'] * 100, 1)}%), "
                    f"rating {p.get('avg_rating')} from {p.get('rating_count')} reviews."
                ),
            }

        if name == "cart_savings_summary":
            cart = cart_service.get_cart(user_id)
            return {
                "item_count": cart["item_count"],
                "subtotal": cart["subtotal"],
                "savings": cart["savings"],
                "items": [
                    {
                        "product_id": i["product_id"],
                        "title": (i.get("title") or "")[:60],
                        "quantity": i["quantity"],
                        "price": i.get("price"),
                        "list_price": i.get("list_price"),
                    }
                    for i in cart["items"]
                ],
            }

        if name == "suggest_next_questions":
            shown = turn.get("shown") or []
            suggestions = []
            if shown:
                title = (shown[0].get("title") or "this item")[:40]
                suggestions.append(f"Why is {title} a good deal?")
                suggestions.append("Add the top one to my cart")
                cat = shown[0].get("category") or "this category"
                suggestions.append(f"Show cheaper options in {cat}")
            else:
                suggestions = [
                    "What are the best deals right now?",
                    "What's in my cart and how much am I saving?",
                    "Build a $100 back-to-school kit",
                ]
            return {"suggestions": suggestions[:3]}

        if name == "find_cheaper_match":
            result = deals.find_cheaper_match(
                product_id=args.get("product_id"),
                query=args.get("query"),
                limit=max(1, min(_to_int(args.get("limit"), 5), 8)),
                retriever=self.retriever,
            )
            products = deals.get_products([p["product_id"] for p in result.get("products") or []])
            if products:
                turn["shown"] = products
            turn["context"].add("Cheaper match")
            return result

        if name == "explain_substitute":
            result = deals.explain_substitute(
                _to_int(args.get("product_id"), -1),
                _to_int(args.get("alternative_id"), -1),
            )
            if "error" not in result:
                turn["shown"] = deals.get_products(
                    [
                        result["original"]["product_id"],
                        result["alternative"]["product_id"],
                    ]
                )
                turn["context"].add("Substitute receipt")
            return result

        if name == "plan_budget_kit":
            kit = budget_kit.plan_kit(
                user_id,
                _to_float(args.get("budget")) or 0,
                list(args.get("needs") or []),
                category=args.get("category") or None,
                retriever=self.retriever,
            )
            pids = [i["product_id"] for i in kit.get("items") or []]
            if pids:
                turn["shown"] = deals.get_products(pids)
            turn["context"].add(f"Budget ${kit.get('budget')}")
            return kit

        if name == "watch_deal":
            entry = deal_watch.watch_product(
                user_id,
                _to_int(args.get("product_id"), -1),
                target_price=_to_float(args.get("target_price")),
            )
            turn["context"].add("Deal watch")
            return {"status": "watching", "watch": entry}

        if name == "get_deal_alerts":
            alerts = deal_watch.list_alerts(user_id, limit=10)
            turn["context"].add("Deal alerts")
            return {"alerts": alerts, "count": len(alerts)}

        if name == "get_member_prefs":
            prefs = member_prefs.get_prefs(user_id)
            return {"prefs": prefs, "summary": member_prefs.format_prefs_for_prompt(prefs)}

        if name == "update_member_prefs":
            patch = {
                k: args[k]
                for k in ("preferred_brands", "avoid_brands", "sizes", "max_budget", "notes")
                if k in args
            }
            prefs = member_prefs.update_prefs(user_id, patch)
            turn["context"].add("Prefs updated")
            return {"prefs": prefs, "summary": member_prefs.format_prefs_for_prompt(prefs)}

        if name == "get_checkout_links":
            handoff = checkout_service.cart_checkout(user_id)
            turn["context"].add("Checkout handoff")
            return handoff

        if name == "control_site":
            from recommenders.site_actions import apply_server_actions, normalize_actions, public_actions

            actions = normalize_actions(args if isinstance(args, dict) else {})
            notes = apply_server_actions(user_id, actions)
            visible = public_actions(actions)
            turn.setdefault("ui_actions", []).extend(visible)
            if visible:
                turn["context"].add("Site action")
            return {"status": "done" if actions else "no_valid_actions", "actions": visible, "notes": notes}

        return {"error": f"Unknown tool '{name}'"}

    def _system_prompt(self, user_id, message: str = ""):
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
        memory_chunks = []
        if message:
            try:
                memory_chunks = chat_memory.retrieve_memory(user_id, message, top_k=3)
            except Exception as e:
                print(f"⚠️ chat RAG retrieve skipped: {e}")
        memory_text = chat_memory.format_memory_for_prompt(memory_chunks)
        try:
            prefs_text = member_prefs.format_prefs_for_prompt(member_prefs.get_prefs(user_id))
        except Exception:
            prefs_text = "unavailable"
        return SYSTEM_PROMPT.format(
            categories=", ".join(categories),
            profile=profile,
            prefs=prefs_text,
            cart=cart_text,
            memory=memory_text,
        )

    def _fallback(self, message, note):
        products = deals.find_best_deals(message, limit=PRODUCTS_SHOWN, retriever=self.retriever)
        return {
            "response": note,
            "products": products,
            "active_context": ["Best value"],
            "pending_actions": [],
            "tools_used": ["find_best_deals"],
            "orchestrator": "fallback",
            "ui_actions": [],
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
            "response": answer or "Done.",
            "products": [{**p, "algorithm": "shopping_agent"} for p in shown if isinstance(p, dict) and p.get("product_id") is not None],
            "active_context": sorted(turn.get("context") or []),
            "pending_actions": turn.get("pending") or [],
            "tools_used": turn.get("tools") or [],
            "ui_actions": list(turn.get("ui_actions") or []),
        }

    # ---------- LangGraph path ----------

    def _build_langchain_tools(self):
        """Tools read user_id/turn from contextvars so the graph can be process-cached."""
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

        class RecallArgs(BaseModel):
            query: str = Field(..., description="What to recall from past chats")
            limit: int = Field(3, description="1-5 memories")

        class WishlistArgs(BaseModel):
            product_id: int
            action: str = Field("add", description="add or remove")

        class ProductArgs(BaseModel):
            product_id: int

        class CheaperArgs(BaseModel):
            product_id: Optional[int] = Field(None, description="Anchor product to beat")
            query: Optional[str] = Field(None, description="Description or image caption")
            limit: int = Field(5, description="1-8 results")

        class SubstituteArgs(BaseModel):
            product_id: int = Field(..., description="Original product")
            alternative_id: int = Field(..., description="Substitute product")

        class BudgetArgs(BaseModel):
            budget: float = Field(..., description="Max spend USD")
            needs: list[str] = Field(..., description="Item needs")
            category: Optional[str] = Field(None)

        class WatchArgs(BaseModel):
            product_id: int
            target_price: Optional[float] = Field(None)

        class PrefsArgs(BaseModel):
            preferred_brands: Optional[list[str]] = None
            avoid_brands: Optional[list[str]] = None
            sizes: Optional[dict] = None
            max_budget: Optional[float] = None
            notes: Optional[str] = None

        class SiteAction(BaseModel):
            type: str
            section: Optional[str] = None
            query: Optional[str] = None
            category: Optional[str] = None
            algo: Optional[str] = None
            product_id: Optional[int] = None
            quantity: Optional[int] = None

        class SiteArgs(BaseModel):
            actions: list[SiteAction] = Field(..., description="Ordered website actions")

        def _call(name, **kwargs):
            ctx = _turn_ctx.get()
            turn = ctx["turn"]
            user_id = ctx["user_id"]
            turn["tools"].append(name)
            try:
                result = self._run_tool(name, kwargs, user_id, turn)
            except Exception as e:
                result = {"error": str(e)[:200]}
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
            StructuredTool.from_function(
                name="recall_past_chats",
                description="RAG search over this member's past chat turns.",
                func=lambda **kw: _call("recall_past_chats", **kw),
                args_schema=RecallArgs,
            ),
            StructuredTool.from_function(
                name="get_wishlist",
                description="List wishlist products.",
                func=lambda **kw: _call("get_wishlist", **kw),
                args_schema=EmptyArgs,
            ),
            StructuredTool.from_function(
                name="update_wishlist",
                description="Add or remove a wishlist product immediately.",
                func=lambda **kw: _call("update_wishlist", **kw),
                args_schema=WishlistArgs,
            ),
            StructuredTool.from_function(
                name="explain_deal",
                description="Explain savings and deal quality for a product.",
                func=lambda **kw: _call("explain_deal", **kw),
                args_schema=ProductArgs,
            ),
            StructuredTool.from_function(
                name="cart_savings_summary",
                description="Cart subtotal and savings versus list price.",
                func=lambda **kw: _call("cart_savings_summary", **kw),
                args_schema=EmptyArgs,
            ),
            StructuredTool.from_function(
                name="suggest_next_questions",
                description="Suggest 3 short follow-up questions.",
                func=lambda **kw: _call("suggest_next_questions", **kw),
                args_schema=EmptyArgs,
            ),
            StructuredTool.from_function(
                name="find_cheaper_match",
                description="Find cheaper substitutes for a product or image/text description.",
                func=lambda **kw: _call("find_cheaper_match", **kw),
                args_schema=CheaperArgs,
            ),
            StructuredTool.from_function(
                name="explain_substitute",
                description="Receipt explaining why a substitute is better or worse.",
                func=lambda **kw: _call("explain_substitute", **kw),
                args_schema=SubstituteArgs,
            ),
            StructuredTool.from_function(
                name="plan_budget_kit",
                description="Pack a multi-item kit under a dollar budget.",
                func=lambda **kw: _call("plan_budget_kit", **kw),
                args_schema=BudgetArgs,
            ),
            StructuredTool.from_function(
                name="watch_deal",
                description="Watch a product for price drops.",
                func=lambda **kw: _call("watch_deal", **kw),
                args_schema=WatchArgs,
            ),
            StructuredTool.from_function(
                name="get_deal_alerts",
                description="List recent price-drop alerts.",
                func=lambda **kw: _call("get_deal_alerts", **kw),
                args_schema=EmptyArgs,
            ),
            StructuredTool.from_function(
                name="get_member_prefs",
                description="Read structured member prefs.",
                func=lambda **kw: _call("get_member_prefs", **kw),
                args_schema=EmptyArgs,
            ),
            StructuredTool.from_function(
                name="update_member_prefs",
                description="Update brands, sizes, budget prefs.",
                func=lambda **kw: _call("update_member_prefs", **kw),
                args_schema=PrefsArgs,
            ),
            StructuredTool.from_function(
                name="get_checkout_links",
                description="Amazon/store checkout deep links for the cart.",
                func=lambda **kw: _call("get_checkout_links", **kw),
                args_schema=EmptyArgs,
            ),
            StructuredTool.from_function(
                name="control_site",
                description=(
                    "Drive the website: navigate, search, filter, switch algorithms, open cart, "
                    "checkout, show a product, summarize reviews, or apply an explicit cart change."
                ),
                func=lambda **kw: _call("control_site", **kw),
                args_schema=SiteArgs,
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

    def _get_graph(self):
        """Compile the ReAct agent once per process (stateless Cloud Run replicas)."""
        if ShoppingAgent._graph is not None:
            return ShoppingAgent._graph
        with ShoppingAgent._graph_lock:
            if ShoppingAgent._graph is not None:
                return ShoppingAgent._graph
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            model = self._langgraph_model()
            tools = self._build_langchain_tools()
            # In-process checkpointer for multi-step tool loops; durable resume
            # across replicas is via Firestore ThreadStore (agent_threads).
            ShoppingAgent._graph = create_react_agent(model, tools, checkpointer=MemorySaver())
            return ShoppingAgent._graph

    def _thread_id(self, user_id) -> str:
        return f"user_{user_id}"

    def _history_messages(self, user_id, conversation_history):
        from langchain_core.messages import AIMessage, HumanMessage

        stored = get_thread_store().load_messages(self._thread_id(user_id))
        source = stored or conversation_history or []
        out = []
        for msg in source[-12:]:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user" and isinstance(content, str):
                out.append(HumanMessage(content=content))
            elif role == "assistant" and isinstance(content, str):
                out.append(AIMessage(content=content))
        return out

    def _persist_thread(self, user_id, message, answer, product_ids=None):
        thread_id = self._thread_id(user_id)
        prior = get_thread_store().load_messages(thread_id)
        prior.append({"role": "user", "content": message})
        prior.append({"role": "assistant", "content": answer})
        try:
            get_thread_store().save_messages(thread_id, prior)
        except Exception as e:
            print(f"⚠️ thread persist skipped: {e}")
        chat_memory.remember_turn(user_id, message, answer, product_ids=product_ids or [])

    def _chat_langgraph(self, user_id, message, conversation_history=None):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        turn = {"shown": [], "pending": [], "context": set(), "tools": []}
        token = _turn_ctx.set({"user_id": user_id, "turn": turn})
        try:
            try:
                graph = self._get_graph()
            except Exception as e:
                print(f"⚠️ LangGraph init failed: {e}")
                if self.clients:
                    return self._chat_legacy(user_id, message, conversation_history)
                return self._fallback(message, "I'm having trouble starting the shopping agent, so here are the best-value matches.")

            thread_id = self._thread_id(user_id)
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": MAX_TOOL_ROUNDS * 2 + 2,
            }

            messages = []
            has_state = False
            try:
                state = graph.get_state(config)
                has_state = bool((state.values or {}).get("messages")) if state else False
            except Exception:
                has_state = False

            # Always refresh system prompt with RAG memory for this query.
            sys = SystemMessage(content=self._system_prompt(user_id, message))
            if not has_state:
                messages.append(sys)
                messages.extend(self._history_messages(user_id, conversation_history))
            else:
                messages.append(sys)
            messages.append(HumanMessage(content=message))

            try:
                result = graph.invoke({"messages": messages}, config=config)
            except Exception as e:
                print(f"⚠️ LangGraph invoke failed: {e}")
                if turn["shown"] or turn["pending"]:
                    out = self._finalize("Here's what I found.", turn)
                    out["orchestrator"] = "langgraph"
                    out["thread_id"] = thread_id
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
            self._persist_thread(
                user_id,
                message,
                answer,
                product_ids=[p["product_id"] for p in out.get("products") or []],
            )
            out["orchestrator"] = "langgraph"
            out["thread_id"] = thread_id
            self._publish_turn(user_id, message, out)
            return out
        finally:
            _turn_ctx.reset(token)

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

        messages = [{"role": "system", "content": self._system_prompt(user_id, message)}]
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
                except (cart_service.CartError, ValueError, LookupError) as e:
                    result = {"error": str(e)}
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})

        if not answer:
            if turn["shown"] or turn["pending"]:
                answer = "Here's what I found."
            else:
                return self._fallback(message, "I'm having trouble reaching the AI agent, so here are the best-value matches for your search.")

        out = self._finalize(answer, turn)
        self._persist_thread(
            user_id,
            message,
            answer,
            product_ids=[p["product_id"] for p in out.get("products") or []],
        )
        out["orchestrator"] = "legacy"
        self._publish_turn(user_id, message, out)
        return out

    def _publish_turn(self, user_id, message, out):
        try:
            from recommenders.events import get_publisher
            get_publisher().agent_turn(
                user_id,
                message,
                out.get("tools_used") or [],
                len(out.get("pending_actions") or []),
                orchestrator=out.get("orchestrator") or "",
            )
        except Exception as e:
            print(f"⚠️ agent turn publish skipped: {e}")

    def chat(self, user_id, message, conversation_history=None):
        try:
            if self.use_langgraph:
                out = self._chat_langgraph(user_id, message, conversation_history)
            else:
                out = self._chat_legacy(user_id, message, conversation_history)
        except Exception as e:
            print(f"⚠️ shopping agent chat failed: {str(e)[:300]}")
            try:
                out = self._fallback(message or "", "I couldn't finish that step, so here are the closest matches.")
            except Exception:
                out = {
                    "response": "I couldn't finish that step. Try a shorter request, like a search or a cart change.",
                    "products": [],
                }
        return self._safe_out(out)

    def _safe_out(self, out):
        """Always return a JSON-safe chat payload, even when a tool or model fails."""
        if not isinstance(out, dict):
            out = {}
        response = out.get("response")
        if not isinstance(response, str) or not response.strip():
            response = "Tell me what to open, search, or change and I will do it on the page."

        def _plain(value):
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, dict):
                return {str(k): _plain(v) for k, v in list(value.items())[:40]}
            if isinstance(value, (list, tuple)):
                return [_plain(v) for v in list(value)[:20]]
            return str(value)[:200]

        products = []
        for product in out.get("products") or []:
            if isinstance(product, dict) and product.get("product_id") is not None:
                products.append(_plain(product))
        return {
            "response": response.strip()[:4000],
            "products": products[:8],
            "active_context": [str(x)[:80] for x in (out.get("active_context") or []) if x][:12],
            "pending_actions": [_plain(a) for a in (out.get("pending_actions") or []) if isinstance(a, dict)][:5],
            "tools_used": [str(t)[:60] for t in (out.get("tools_used") or [])][:20],
            "orchestrator": str(out.get("orchestrator") or "")[:40] or None,
            "thread_id": out.get("thread_id") if isinstance(out.get("thread_id"), str) else None,
            "ui_actions": [_plain(a) for a in (out.get("ui_actions") or []) if isinstance(a, dict)][:8],
        }
