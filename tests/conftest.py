import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent

chatbot_src = root_dir / "src" / "apis" / "chatbot" / "src"
docparser_src = root_dir / "src" / "apis" / "docparser" / "src"

sys.path.insert(0, str(chatbot_src))
sys.path.insert(0, str(docparser_src))