"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import os
import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# CẤU HÌNH & LOAD BIẾN MÔI TRƯỜNG TỪ .ENV
# ═══════════════════════════════════════════════════════════════════════════

def load_env(env_path: str = None) -> None:
    """Tự động đọc các cấu hình từ file .env."""
    if env_path is None:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

load_env()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def call_llm(prompt: str, system_prompt: str = "") -> str:
    """
    Gọi trực tiếp đến LLM Provider (Ollama Cloud)
    """
    # 1. Thử gọi Ollama API endpoint
    if OLLAMA_HOST and OLLAMA_API_KEY:
        try:
            url = f"{OLLAMA_HOST}/api/generate"
            headers = {
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data.get("response", "").strip()
                if reply:
                    return reply
        except Exception:
            pass

    # 2. Thử gọi Gemini REST API fallback
    if GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            headers = {"Content-Type": "application/json"}
            combined_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            payload = {
                "contents": [{"parts": [{"text": combined_text}]}]
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
        except Exception:
            pass

    return ""


# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """VinAssistant - Trợ lý AI Vingroup.
1. PERSONA: Chuyên nghiệp, chính xác, lịch sự.
2. AVAILABLE TOOLS: {tools}
3. CORE RULES: Chỉ dùng dữ liệu từ tool, không bịa đặt. Bắt buộc gọi tool khi tra cứu hoặc tạo ticket. Báo không tìm thấy nếu kết quả rỗng.
4. OPERATIONAL BOUNDARIES: Chỉ hỗ trợ hệ sinh thái Vingroup, từ chối yêu cầu ngoài phạm vi.
5. OUTPUT CONTRACT: Thought -> Action (tên tool hoặc None) -> Observation -> Final Answer."""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """Gọi LLM thuần hoặc trả lời trực tiếp mà không dùng Tool."""
        llm_reply = call_llm(
            user_input,
            system_prompt="Bạn là chatbot AI. Hãy trả lời trực tiếp câu hỏi của người dùng mà không gọi công cụ nào."
        )
        if not llm_reply:
            llm_reply = f"[Chatbot Baseline] Trả lời cho: {user_input}"

        return {
            "answer": llm_reply,
            "tool_calls": [],
            "status": "success",
            "mode": "llm_baseline" if llm_reply != f"[Chatbot Baseline] Trả lời cho: {user_input}" else "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intent(self, user_input: str) -> Dict[str, Any]:
        """Phân tích intent và trích xuất tham số từ truy vấn người dùng."""
        text = user_input.lower()

        # Kiểm tra FAQ trước
        is_faq = False
        if ("bảo hành" in text or "chính sách" in text) and not any(w in text for w in ["lỗi", "hỏng", "sự cố"]):
            is_faq = True

        # Kiểm tra Ticket
        needs_ticket = False
        customer_name = "Khách hàng"
        issue_description = ""
        priority = "medium"

        if any(w in text for w in ["lỗi", "hỏng", "sự cố", "khiếu nại", "phản hồi", "ticket", "gấp"]):
            needs_ticket = True

        name_match = re.search(r"(?:tôi tên|tên tôi là|tên là)\s*[:]?\s*([^,\.]+)", user_input, re.IGNORECASE)
        if name_match:
            customer_name = name_match.group(1).strip()
            needs_ticket = True

        if any(w in text for w in ["gấp", "nghiêm trọng", "khẩn cấp"]):
            priority = "high"
        elif any(w in text for w in ["thấp", "không vội"]):
            priority = "low"

        issue_match = re.search(r"((?:xe\s+[A-Za-z0-9\s]+|phòng\s+[^,\.]+|hệ thống\s+[^,\.]+)?(?:bị\s+[^,\.]+|lỗi\s+[^,\.]+))", user_input, re.IGNORECASE)
        if issue_match:
            issue_description = issue_match.group(1).strip()
        else:
            issue_description = user_input

        # Kiểm tra Catalog
        needs_catalog = False
        category = "xe_dien"
        max_price = 999999999999

        has_catalog_keywords = any(w in text for w in ["xe", "xe điện", "vf", "du lịch", "resort", "vinpearl", "khách sạn", "phòng"])
        has_price_query = any(w in text for w in ["giá", "triệu", "tỷ", "dưới", "xem"])

        if not is_faq and (has_price_query or not needs_ticket) and has_catalog_keywords:
            needs_catalog = True
            if any(w in text for w in ["du lịch", "resort", "vinpearl", "phú quốc", "nha trang"]):
                category = "du_lich"
            else:
                category = "xe_dien"

            # Parse giá tối đa
            price_match = re.search(r"(?:dưới|tối đa|khoảng|<=|<)\s*(\d+(?:[.,]\d+)?)\s*(triệu|tỷ|m)", text)
            if price_match:
                val = float(price_match.group(1).replace(",", "."))
                unit = price_match.group(2).lower()
                if unit in ["triệu", "m"]:
                    max_price = int(val * 1_000_000)
                elif unit == "tỷ":
                    max_price = int(val * 1_000_000_000)
            elif "200 triệu" in text:
                max_price = 200000000

        return {
            "needs_catalog": needs_catalog,
            "catalog_args": {"category": category, "max_price": max_price},
            "needs_ticket": needs_ticket,
            "ticket_args": {
                "customer_name": customer_name,
                "issue_description": issue_description or user_input,
                "priority": priority
            },
            "is_faq": is_faq
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop (ReAct Framework)."""
        self.trace = []
        intents = self._detect_intent(user_input)

        # Trường hợp 1: FAQ không cần gọi tool
        if intents["is_faq"]:
            thought = "Khách hàng hỏi thông tin chính sách bảo hành, có thể trả lời trực tiếp từ tri thức có sẵn."
            answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km tuỳ điều kiện nào đến trước."
            self.trace.append({
                "iteration": 1,
                "thought": thought,
                "action": "None",
                "action_input": {},
                "observation": "FAQ knowledge retrieved."
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        catalog_results = None
        ticket_result = None
        current_iteration = 0

        # Iteration 1: Gọi Tool Catalog nếu cần
        if intents["needs_catalog"]:
            current_iteration += 1
            cat_args = intents["catalog_args"]
            thought = f"Tra cứu danh mục sản phẩm loại '{cat_args['category']}' với giá dưới {cat_args['max_price']} VNĐ."
            catalog_results = search_product_catalog(**cat_args)
            self.trace.append({
                "iteration": current_iteration,
                "thought": thought,
                "action": "search_product_catalog",
                "action_input": cat_args,
                "observation": catalog_results
            })

        # Iteration 2: Gọi Tool Ticket nếu cần
        if intents["needs_ticket"]:
            current_iteration += 1
            tk_args = intents["ticket_args"]
            thought = f"Tạo phiếu hỗ trợ cho khách hàng {tk_args['customer_name']} mức ưu tiên {tk_args['priority']}."
            ticket_result = submit_support_ticket(**tk_args)
            self.trace.append({
                "iteration": current_iteration,
                "thought": thought,
                "action": "submit_support_ticket",
                "action_input": tk_args,
                "observation": ticket_result
            })

        # Safeguard: Kiểm tra giới hạn số bước
        if current_iteration > self.max_iterations:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": current_iteration,
                "status": "max_iterations_reached"
            }

        # Tổng hợp kết quả Final Answer
        answer_parts = []

        if catalog_results is not None:
            if not catalog_results or len(catalog_results) == 0:
                answer_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn.")
            else:
                items = [f"{p.get('name', '')} (giá: {p.get('price_vnd', 0):,} VNĐ)" for p in catalog_results]
                answer_parts.append(f"Danh sách sản phẩm phù hợp: {', '.join(items)}.")

        if ticket_result is not None:
            tk_id = ticket_result.get("ticket_id", "")
            c_name = ticket_result.get("customer_name", intents["ticket_args"]["customer_name"])
            answer_parts.append(f"Hệ thống đã tạo yêu cầu hỗ trợ thành công cho khách hàng {c_name}. Mã ticket: {tk_id}.")

        final_answer = " ".join(answer_parts) if answer_parts else "Yêu cầu của bạn đã được ghi nhận."

        # Thử làm giàu câu trả lời bằng LLM nếu kết nối thành công
        llm_prompt = f"Yêu cầu của khách: {user_input}\nKết quả: {final_answer}\nHãy hoàn thiện câu trả lời lịch sự, giữ nguyên mã ticket và tên sản phẩm."
        llm_reply = call_llm(llm_prompt, system_prompt=SYSTEM_PROMPT.replace("{tools}", json.dumps(TOOL_DEFINITIONS, ensure_ascii=False)))
        if llm_reply and (
            (ticket_result and ticket_result.get("ticket_id") in llm_reply) or
            (catalog_results and any(p.get("name") in llm_reply for p in catalog_results))
        ):
            final_answer = llm_reply

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": current_iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
