import asyncio
import json
import logging
import os
import re
from pathlib import Path

import discord
from anthropic import APIError, AsyncAnthropic
from anthropic._exceptions import APIConnectionError, APIStatusError, OverloadedError, RateLimitError
from dotenv import load_dotenv

from personality import L_SYSTEM_PROMPT

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
HISTORY_LIMIT = 10
DISCORD_MSG_LIMIT = 2000
CONTINUATION_WINDOW_SEC = 90

NAME_PATTERN = re.compile(r"(?:^|[^가-힣ㄱ-ㅎㅏ-ㅣ])엘(?:[^가-힣ㄱ-ㅎㅏ-ㅣ]|$)")
USAGE_PATTERN = re.compile(r"(?:^|\s)/usage\b")

USAGE_FILE = Path(__file__).parent / "_usage.json"
CHAP_PATH = Path(__file__).parent / "assets" / "챱.png"

# $/MTok (input, output). web_search 검색 비용은 별도 ($0.01/검색).
MODEL_RATES = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
}


def load_usage() -> dict:
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"requests": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_creation_tokens": 0, "cache_read_tokens": 0}


def save_usage(u: dict) -> None:
    USAGE_FILE.write_text(json.dumps(u, indent=2))


def record_usage(usage) -> None:
    u = load_usage()
    u["requests"] += 1
    u["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
    u["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
    u["cache_creation_tokens"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
    u["cache_read_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    save_usage(u)


def format_usage() -> str:
    u = load_usage()
    in_rate, out_rate = MODEL_RATES.get(MODEL, (3.0, 15.0))
    cost = (u["input_tokens"] / 1_000_000) * in_rate + (u["output_tokens"] / 1_000_000) * out_rate
    return (
        f"엘이 지금까지 소비한 자원입니다.\n"
        f"```\n"
        f"모델       : {MODEL}\n"
        f"요청 수    : {u['requests']:,}\n"
        f"입력 토큰  : {u['input_tokens']:,}\n"
        f"출력 토큰  : {u['output_tokens']:,}\n"
        f"캐시 읽기  : {u['cache_read_tokens']:,}\n"
        f"예상 비용  : ${cost:.4f}\n"
        f"```\n"
        f"(web_search 호출은 검색당 약 $0.01 별도)"
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("L")

intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
client = discord.Client(intents=intents)
anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


def strip_mention(text: str, bot_id: int) -> str:
    return re.sub(rf"<@!?{bot_id}>", "", text).strip()


async def collect_history(msg: discord.Message) -> list[dict]:
    chain: list[discord.Message] = [msg]

    current = msg
    while len(chain) < HISTORY_LIMIT:
        ref = current.reference
        if not ref or not ref.message_id:
            break
        try:
            parent = ref.cached_message or await msg.channel.fetch_message(ref.message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            break
        chain.append(parent)
        current = parent

    if len(chain) == 1:
        try:
            async for prev in msg.channel.history(limit=HISTORY_LIMIT - 1, before=msg):
                age = (msg.created_at - prev.created_at).total_seconds()
                if age > 300:
                    break
                chain.append(prev)
        except discord.HTTPException:
            pass

    chain.reverse()

    bot_id = client.user.id if client.user else 0
    messages: list[dict] = []
    for m in chain:
        content = strip_mention(m.content, bot_id) if client.user else m.content
        if not content:
            continue
        if m.author.id == bot_id:
            role = "assistant"
            line = content
        else:
            role = "user"
            display = getattr(m.author, "display_name", None) or m.author.name
            line = f"[{display}] {content}"
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + line
        else:
            messages.append({"role": role, "content": line})

    if not messages or messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "(빈 메시지)"})
    return messages


def extract_text(response) -> str:
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(p for p in parts if p).strip()


def split_for_discord(text: str) -> list[str]:
    if len(text) <= DISCORD_MSG_LIMIT:
        return [text]
    chunks = []
    remaining = text
    while len(remaining) > DISCORD_MSG_LIMIT:
        cut = remaining.rfind("\n", 0, DISCORD_MSG_LIMIT)
        if cut == -1:
            cut = remaining.rfind(" ", 0, DISCORD_MSG_LIMIT)
        if cut == -1:
            cut = DISCORD_MSG_LIMIT
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


async def is_continuation(msg: discord.Message) -> bool:
    """직전에 봇이 말했고, 90초 이내, 같은 사용자(또는 봇만 있던 흐름)일 때 True."""
    bot_id = client.user.id if client.user else 0
    try:
        async for prev in msg.channel.history(limit=5, before=msg):
            age = (msg.created_at - prev.created_at).total_seconds()
            if age > CONTINUATION_WINDOW_SEC:
                return False
            if prev.author.id == bot_id:
                return True
            if prev.author.id == msg.author.id:
                continue
            return False
    except discord.HTTPException:
        return False
    return False


async def should_respond(msg: discord.Message) -> bool:
    if msg.author.bot or (client.user and msg.author.id == client.user.id):
        return False
    if isinstance(msg.channel, discord.DMChannel):
        return True
    if client.user in msg.mentions:
        return True
    if NAME_PATTERN.search(msg.content):
        return True
    ref = msg.reference
    if ref and ref.message_id:
        try:
            parent = ref.cached_message or await msg.channel.fetch_message(ref.message_id)
            if client.user and parent.author.id == client.user.id:
                return True
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            pass
    if await is_continuation(msg):
        return True
    return False


@client.event
async def on_ready():
    log.info("Logged in as %s (id=%s)", client.user, client.user.id if client.user else "?")


SYSTEM_BLOCKS = [
    {
        "type": "text",
        "text": L_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


async def call_claude(history: list[dict]):
    """Anthropic 호출. 시스템 프롬프트 캐싱 적용 + 일시 과부하/레이트리밋 지수 백오프 재시도."""
    max_attempts = 5
    delay = 1.5
    for attempt in range(max_attempts):
        try:
            return await anthropic.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_BLOCKS,
                messages=history,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
            )
        except (OverloadedError, RateLimitError, APIConnectionError) as e:
            if attempt == max_attempts - 1:
                raise
            log.warning("Transient error %s, retrying in %.1fs (attempt %d/%d)",
                        type(e).__name__, delay, attempt + 1, max_attempts)
            await asyncio.sleep(delay)
            delay *= 2


@client.event
async def on_message(msg: discord.Message):
    if not await should_respond(msg):
        return

    bot_id = client.user.id if client.user else 0
    stripped = strip_mention(msg.content, bot_id)
    ch_name = getattr(msg.channel, "name", "DM")
    preview = msg.content.replace("\n", " ")[:60]
    log.info("EVENT trigger user=%s ch=%s text=%r", msg.author, ch_name, preview)

    try:
        await msg.add_reaction("👀")
    except (discord.Forbidden, discord.HTTPException):
        pass

    if client.user in msg.mentions and "챱" in stripped:
        log.info("EVENT chap_trigger")
        try:
            await msg.channel.send(file=discord.File(str(CHAP_PATH)))
        except FileNotFoundError:
            await msg.channel.send("...챱이 사라졌습니다.")
        except discord.HTTPException:
            log.exception("EVENT chap_send_fail")
            await msg.channel.send("...챱을 보내려다 실패했습니다.")
        return

    if USAGE_PATTERN.search(stripped):
        log.info("EVENT usage_query")
        await msg.channel.send(format_usage())
        return

    history = await collect_history(msg)
    log.info("EVENT history msgs=%d", len(history))

    async with msg.channel.typing():
        try:
            response = await call_claude(history)
        except (APIStatusError, APIError):
            log.exception("EVENT api_error")
            await msg.channel.send("...엘이 잠시 사고가 멈춘 모양입니다. 다시 불러주세요.")
            return

    record_usage(response.usage)
    log.info("EVENT response in=%d out=%d stop=%s",
             getattr(response.usage, "input_tokens", 0),
             getattr(response.usage, "output_tokens", 0),
             response.stop_reason)

    text = extract_text(response)
    if not text:
        log.info("EVENT empty_text")
        await msg.channel.send("...(엘이 잠시 생각에 잠겼습니다.)")
        return

    for chunk in split_for_discord(text):
        await msg.channel.send(chunk)
    log.info("EVENT sent chunks=%d chars=%d", len(split_for_discord(text)), len(text))


if __name__ == "__main__":
    client.run(DISCORD_TOKEN, log_handler=None)
