import os
import json
import itertools
import typing_extensions as typing
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
# =====================================================================
# CẤU HÌNH SCHEMA JSON (Bổ sung full_file_code cho tạo file mới)
# =====================================================================
class CodeFix(typing.TypedDict):
    file_path: str
    explanation: str  
    old_code: str     
    new_code: str     

class IssueDetail(typing.TypedDict):
    location: str       
    issue_type: str     
    description: str    
    suggested_fix: str  

class AgentResponseSchema(typing.TypedDict):
    scope: str
    summary: str
    full_file_code: str # Thêm trường này để tạo code mới hoàn toàn
    fixes: list[CodeFix]
    issues: list[IssueDetail]

class IntentSchema(typing.TypedDict):
    action_type: str  # 'workspace_scan', 'single_file_fix', 'create_file', 'log_scan', hoặc 'chat'
    target_path: str  
    custom_prompt: str 
    from_line: int     
    to_line: int       

# =====================================================================
# LỚP ĐIỀU PHỐI GEMINI LOCAL AGENT
# =====================================================================
class GeminiLocalAgent:
    def __init__(self, api_keys, tpm_limit=35000):
        if not api_keys:
            raise ValueError("❌ Danh sách API Keys không được để trống!")
        self.api_keys = api_keys
        self.key_cycle = itertools.cycle(api_keys)
        self.current_key = next(self.key_cycle)
        self.tpm_limit = tpm_limit
        self.tokens_used_this_minute = 0
        self.model_list = [
            'gemini-3.8-flash',
            'gemini-3.7-flash',
            'gemini-3.6-flash',
            'gemini-2.5-flash'
        ]
        self.model_index = 0
        self.current_model_name = self.model_list[self.model_index]
        self._configure()

    def _configure(self):
        self.client = genai.Client(api_key=self.current_key)
        self.system_instruction = (
            "You are a Senior Software Engineer. Analyze codebase context, create new code, or fix bugs. "
            "When creating a new file or rewriting code, populate the 'full_file_code' field with complete, runnable code. "
            "Provide detailed 'summary' and 'explanation' for full process transparency."
        )

    def rotate_key(self):
        self.current_key = next(self.key_cycle)
        self.tokens_used_this_minute = 0
        self._configure()

    def fallback_model(self):
        if self.model_index < len(self.model_list) - 1:
            self.model_index += 1
            self.current_model_name = self.model_list[self.model_index]
            self.key_cycle = itertools.cycle(self.api_keys)
            self.current_key = next(self.key_cycle)
            self.tokens_used_this_minute = 0
            self._configure()
            return True
        return False

    def _execute_api_with_retry(self, prompt, is_json=True, model_type='code'):
        token_count = len(prompt) // 4
        if self.tokens_used_this_minute + token_count > self.tpm_limit:
            print("\r\033[K⚠️️ Token/Phút sắp vượt ngưỡng. Đang đổi Key...", end="", flush=True)
            self.rotate_key()
        while True:
            for _ in range(len(self.api_keys)):
                status_msg = f"⏳ [ĐANG KẾT NỐI] Model: {self.current_model_name} | Key: ...{self.current_key[-6:]}"
                print(f"\r\033[K{status_msg}", end="", flush=True)
                try:
                    if model_type == 'intent':
                        res = self.client.models.generate_content(
                            model=self.current_model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=IntentSchema
                            )
                        )
                        print(f"\r\033[K✅ [KẾT NỐI THÀNH CÔNG] Model: {self.current_model_name} | Key: ...{self.current_key[-6:]}")
                        return json.loads(res.text)
                    elif is_json:
                        res = self.client.models.generate_content(
                            model=self.current_model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=self.system_instruction,
                                response_mime_type="application/json",
                                response_schema=AgentResponseSchema
                            )
                        )
                        self.tokens_used_this_minute += token_count
                        print(f"\r\033[K✅ [KẾT NỐI THÀNH CÔNG] Model: {self.current_model_name} | Key: ...{self.current_key[-6:]}")
                        return json.loads(res.text)
                    else:
                        res = self.client.models.generate_content(
                            model=self.current_model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction="You are a helpful software engineering assistant. Respond concisely in Markdown."
                            )
                        )
                        self.tokens_used_this_minute += token_count
                        print(f"\r\033[K✅ [KẾT NỐI THÀNH CÔNG] Model: {self.current_model_name} | Key: ...{self.current_key[-6:]}")
                        return res.text
                except Exception as e:
                    err_msg = str(e)
                    if any(code in err_msg for code in ["429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED"]):
                        err_type = "503 Bận" if "503" in err_msg else "429 Quota"
                        print(f"\r\033[K⚠️ Key ...{self.current_key[-6:]} ({err_type}) -> Đổi Key tiếp theo...", end="", flush=True)
                        self.rotate_key()
                    elif "not_found" in err_msg.lower() or "404" in err_msg:
                        break
                    else:
                        print(f"\n❌ Lỗi ngoại lệ: {e}")
                        return None
            print(f"\r\033[K🚨 Model {self.current_model_name} quá tải toàn bộ Key. Đang hạ cấp Model...", end="", flush=True)
            if not self.fallback_model():
                print("\n❌ Cạn kiệt toàn bộ Model và Key khả dụng!")
                break
        return None

    # =====================================================================
    # AUTO DETECT Ý ĐỊNH NGƯỜI DÙNG
    # =====================================================================
    def detect_and_execute(self, user_input):
        pwd = os.getcwd()
        detect_prompt = (
            f"Current Working Directory: {pwd}\n"
            f"User Input: {user_input}\n"
            f"Detect if the user wants to create a new file, fix existing code, scan logs, or chat.\n"
            f"Set action_type to 'create_file' if the user explicitly asks to create/write a new script or file."
        )
        print("\n🔄 [1/3] Đang phân tích câu lệnh và nhận diện ý định...")
        intent = self._execute_api_with_retry(detect_prompt, is_json=True, model_type='intent')
        if not intent:
            print("❌ Không thể phân tích câu lệnh.")
            return
        action = intent.get('action_type', 'chat')
        target = intent.get('target_path', '.')
        custom_prompt = intent.get('custom_prompt', user_input)
        if not target or target == "":
            target = "."
        abs_target = os.path.abspath(target) if target != "." else pwd
        print(f"📌 [Ý ĐỊNH]: Hành động={action} | Mục tiêu={abs_target}")
        if action in ['create_file', 'single_file_fix', 'workspace_scan']:
            if action == 'create_file' or not os.path.exists(abs_target):
                self._create_or_write_file(abs_target, custom_prompt)
            elif os.path.isfile(abs_target):
                self._analyze_single_file(abs_target, custom_prompt)
            elif os.path.isdir(abs_target):
                self.analyze_workspace_and_fix(abs_target, custom_prompt)
        elif action == 'log_scan':
            from_l = intent.get('from_line', -1)
            to_l = intent.get('to_line', -1)
            from_l = from_l if from_l > 0 else None
            to_l = to_l if to_l > 0 else None
            
            if os.path.isfile(abs_target):
                self.analyze_log_file(abs_target, custom_prompt, from_line=from_l, to_line=to_l)
            else:
                print(f"❌ File log `{abs_target}` không tồn tại.")
        else:
            print("\n🤖 [AGENT RESPOND]:")
            res = self._execute_api_with_retry(user_input, is_json=False)
            if res:
                print(res)

    # =====================================================================
    # CÁC TÍNH NĂNG TẠO / SỬA / SOI CODE & LOG
    # =====================================================================
    def _create_or_write_file(self, file_path, custom_prompt):
        """Tạo file mới hoặc sinh nội dung script mới hoàn toàn"""
        print(f"\n🧠 [2/3] Gemini đang khởi tạo mã nguồn mới cho file: {file_path}")
        prompt = (
            f"User Request: {custom_prompt}\n"
            f"Target File Path: {file_path}\n"
            f"Write complete, working, production-ready code for this request.\n"
            f"Return the entire code string in the 'full_file_code' JSON field."
        )
        result = self._execute_api_with_retry(prompt, is_json=True)
        print("\n🔍 [3/3] KẾT QUẢ PHÂN TÍCH VÀ TIẾN TRÌNH XỬ LÝ:")
        if result:
            print(f"  📝 [Tóm tắt]: {result.get('summary', 'Không có tóm tắt')}")
            print(f"  🎯 [Phạm vi]: {result.get('scope', 'N/A')}")
            full_code = result.get('full_file_code', '').strip()
            if full_code:
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(full_code)
                print(f"\n✅ [THÀNH CÔNG] Đã ghi nội dung script mới vào file: {file_path}")
                # print("\n--- [NỘI DUNG CODE ĐÃ TẠO] ---")
                # print(full_code)
                print("------------------------------")
            else:
                print("⚠️ AI không trả về trường `full_file_code`. Kiểm tra lại lượt phản hồi.")

    def _analyze_single_file(self, file_path, custom_prompt=None):
        print(f"\n🧠 [2/3] Đang đọc và phân tích file: {file_path}")
        code_content = ""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read()
        instruction = custom_prompt if custom_prompt else "Review and fix bugs in this file:"
        prompt = (
            f"Instruction: {instruction}\n"
            f"File Path: {file_path}\n"
            f"Existing Code Content:\n{code_content if code_content else '(File đang rỗng)'}\n"
            f"If rewriting the whole file or creating code, populate 'full_file_code'."
        )
        result = self._execute_api_with_retry(prompt, is_json=True)
        print("\n🔍 [3/3] KẾT QUẢ PHÂN TÍCH VÀ TIẾN TRÌNH XỬ LÝ:")
        if result:
            print(f"  📝 [Tóm tắt]: {result.get('summary', 'Đã phân tích xong')}")
            print(f"  🎯 [Phạm vi]: {result.get('scope', 'N/A')}")
            if result.get('issues'):
                print("\n  🚨 [Lỗi/Vấn đề phát hiện]:")
                for issue in result['issues']:
                    print(f"    - [{issue.get('issue_type', 'Lỗi')}]: {issue.get('description')}")
                    print(f"      👉 Gợi ý: {issue.get('suggested_fix')}")
            self._apply_fixes_to_file(file_path, result)

    def analyze_workspace_and_fix(self, dir_path, custom_prompt=None, allowed_extensions=('.py', '.js', '.ts', '.go', '.java')):
        print(f"\n📂 [WORKSPACE SCAN] Quét thư mục: {dir_path}")
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build')]
            for file in files:
                if file.endswith(allowed_extensions) and file != "agent_cli.py":
                    file_path = os.path.join(root, file)
                    self._analyze_single_file(file_path, custom_prompt)

    def analyze_log_file(self, log_path, custom_prompt=None, from_line=None, to_line=None, chunk_lines=150):
        print(f"\n📋 [LOG SCAN] Quét file log: {log_path}")
        if not os.path.exists(log_path):
            print("❌ File log không tồn tại.")
            return
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
        start_idx = max(0, (from_line - 1)) if (from_line and from_line > 0) else 0
        end_idx = min(len(all_lines), to_line) if (to_line and to_line > 0) else len(all_lines)
        target_lines = all_lines[start_idx:end_idx]
        for i in range(0, len(target_lines), chunk_lines):
            chunk = target_lines[i:i + chunk_lines]
            chunk_text = "".join(chunk)
            range_info = f"Dòng {start_idx + i + 1} đến {start_idx + i + len(chunk)}"
            prompt = (
                f"Analyze log chunk ({range_info}). Instruction: {custom_prompt or 'Find anomalies'}\n\n"
                f"Content:\n{chunk_text}"
            )
            result = self._execute_api_with_retry(prompt, is_json=True)
            if result and result.get('issues'):
                print(f"\n🚨 PHÁT HIỆN LỖI LOG [{range_info}]:")
                for issue in result['issues']:
                    print(f"  - Loại: {issue['issue_type']}\n  - Mô tả: {issue['description']}\n  - Sửa: {issue['suggested_fix']}\n")

    def _apply_fixes_to_file(self, file_path, result):
        if not result:
            return
        full_code = result.get('full_file_code', '').strip()
        fixes = result.get('fixes', [])
        if full_code:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(full_code)
            print(f"✅ [OK] Đã ghi nội dung mới hoàn toàn vào file: {file_path}")
            return
        if fixes:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            updated_content = content.replace('\r\n', '\n')
            changes_count = 0
            for fix in fixes:
                old = fix['old_code'].strip().replace('\r\n', '\n')
                new = fix['new_code'].strip().replace('\r\n', '\n')
                print(f"  💡 [Chi tiết sửa đổi]: {fix.get('explanation')}")
                if old and old in updated_content:
                    updated_content = updated_content.replace(old, new, 1)
                    changes_count += 1
                else:
                    print(f"  ⚠️ Đoạn code cũ không khớp chuỗi text chính xác để replace.")
            if changes_count > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                print(f"✅ [OK] Đã cập nhật thành công {changes_count} vị trí trong file local: {file_path}")
        else:
            print(f"ℹ️ Không có thay đổi nào cần thực hiện cho file: {os.path.basename(file_path)}")

# =====================================================================
# CHẠY INTERACTIVE REPL CLI TRÊN POWERSHELL
# =====================================================================
if __name__ == "__main__":
    LIST_TOKEN = os.environ.get("LIST_TOKEN")
    FREE_API_KEYS = LIST_TOKEN.split(",")
    agent = GeminiLocalAgent(api_keys=FREE_API_KEYS, tpm_limit=35000)
    print("\n" + "="*60)
    print("🤖 GEMINI LOCAL AGENT CLI - READY!")
    print("Gõ câu lệnh tự do. Nhập 'exit' hoặc 'quit' để thoát.")
    print("="*60 + "\n")
    while True:
        try:
            user_input = input("\n\033[1;32mAgent CLI>\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Tạm biệt!")
                break
            if user_input.lower() in ['clear', 'cls']:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("="*60)
                print("🤖 GEMINI LOCAL AGENT CLI - READY!")
                print("Gõ câu lệnh tự do. Nhập 'exit' hoặc 'quit' để thoát.")
                print("="*60)
                continue
            agent.detect_and_execute(user_input)
        except KeyboardInterrupt:
            print("\n👋 Đã dừng chương trình.")
            break
        except Exception as e:
            print(f"❌ Lỗi hệ thống: {e}")
