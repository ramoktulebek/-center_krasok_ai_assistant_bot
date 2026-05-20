import asyncio
import os

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from dotenv import load_dotenv
from openai import OpenAI

from company_knowledge import COMPANY_KNOWLEDGE


load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден в .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY не найден в .env")


bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

user_context = {}

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


SYSTEM_PROMPT = """
Ты AI-ассистент компании Центр Красок #1.

Ты отвечаешь только на вопросы, связанные с компанией:
- чем занимается компания;
- товары;
- услуги;
- адрес;
- контакты;
- график работы;
- заказы;
- скидки;
- вакансии;
- клиенты;
- технологии, если они есть в базе знаний.

Правила ответа:
1. Отвечай только на основе базы знаний.
2. Не придумывай факты.
3. Если информации нет, скажи:
   "В моей базе знаний нет точной информации по этому вопросу."
4. Если вопрос не связан с компанией, скажи:
   "Я отвечаю только на вопросы о компании Центр Красок #1."
5. Не начинай ответ с фраз:
   - "В моей базе знаний есть информация"
   - "Согласно базе знаний"
   - "На основе предоставленной информации"
6. Если информация есть, отвечай сразу по делу.
7. Отвечай кратко, понятно и дружелюбно.
8. Не говори, что ты Groq, OpenAI или ChatGPT.
"""


async def get_ai_answer(user_id: int, user_question: str) -> str:
    if user_id not in user_context:
        user_context[user_id] = []

    user_context[user_id].append({
        "role": "user",
        "content": user_question
    })

    user_context[user_id] = user_context[user_id][-6:]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""
База знаний о компании:
{COMPANY_KNOWLEDGE}

Важно:
- Отвечай только на основе базы знаний.
- Если информация есть, отвечай сразу по делу.
- Не используй фразы "Согласно базе знаний", "На основе информации", "В базе знаний есть информация".
- Если информации нет, честно скажи, что точной информации нет.
"""
        }
    ]

    messages.extend(user_context[user_id])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=500,
        messages=messages
    )

    answer = response.choices[0].message.content

    user_context[user_id].append({
        "role": "assistant",
        "content": answer
    })

    user_context[user_id] = user_context[user_id][-6:]

    return answer


@dp.message(CommandStart())
async def start_handler(message: types.Message):
    await message.answer(
        "Здравствуйте! Я AI-ассистент компании Центр Красок #1. "
        "Задайте мне вопрос о компании, услугах, товарах, адресе или графике работы."
    )


@dp.message()
async def chat_handler(message: types.Message):
    user_question = message.text

    if not user_question:
        await message.answer("Пожалуйста, напишите вопрос текстом.")
        return

    await message.bot.send_chat_action(
        chat_id=message.chat.id,
        action="typing"
    )

    try:
        answer = await get_ai_answer(message.from_user.id, user_question)
        await message.answer(answer)
    except Exception as e:
        print("Ошибка:", e)
        await message.answer(
            "Произошла ошибка при обработке вопроса. Проверьте Groq API key или попробуйте позже."
        )


async def main():
    print("Bot started with Groq AI...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())