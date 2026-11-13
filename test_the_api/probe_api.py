import os
import requests

api_key = os.environ["COURSE_API_KEY"]   # 先在环境种设置 COURSE_API_KEY=你的APIKEY

session = requests.Session()

# 不读取 Windows 环境中的代理配置
session.trust_env = False

response = session.post(
    "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": "Qwen",
        "messages": [
            {"role": "user", "content": "Reply with exactly: API OK"}
        ],
        "max_tokens": 32,
    },
    timeout=30,
)

print("HTTP:", response.status_code)
print("Response:", response.text)