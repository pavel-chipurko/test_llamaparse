"""
Пример запуска обработки документов.

Использование:
    uv run python run_example.py путь/к/файлу.pdf
    uv run python run_example.py фото1.jpg фото2.jpg фото3.jpg
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from extraction_review.clients import get_llama_cloud_client
from extraction_review.process_file import FileEvent, workflow


async def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  uv run python run_example.py документ.pdf")
        print("  uv run python run_example.py фото1.jpg фото2.jpg фото3.jpg")
        return

    files = sys.argv[1:]
    client = await get_llama_cloud_client()

    print(f"\n📁 Загрузка {len(files)} файл(ов)...")

    # Загружаем файлы в LlamaCloud
    file_ids = []
    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            print(f"❌ Файл не найден: {filepath}")
            return

        print(f"   Загружаю: {path.name}")
        uploaded = await client.files.create(
            file=str(path),
            purpose="split",
        )
        file_ids.append(uploaded.id)

    # Запускаем обработку
    print("\n🔍 Анализирую документы...")

    if len(file_ids) == 1:
        event = FileEvent(file_id=file_ids[0])
    else:
        event = FileEvent(file_ids=file_ids)

    result = await workflow.run(start_event=event)

    # Выводим результаты
    print("\n" + "=" * 50)
    print("✅ РЕЗУЛЬТАТЫ ОБРАБОТКИ")
    print("=" * 50)

    if result.merged_from_files > 1:
        print(f"📎 Объединено файлов: {result.merged_from_files}")

    print(f"📄 Всего страниц: {result.total_pages}")
    print(f"📑 Найдено документов: {len(result.segments)}")
    print()

    for i, seg in enumerate(result.segments, 1):
        print(f"{i}. {seg.filename}")
        print(f"   Тип: {seg.category}")
        print(f"   Страницы: {seg.pages}")
        print(f"   Уверенность: {seg.confidence}")
        print(f"   ID файла: {seg.new_file_id}")
        print()

    print("=" * 50)
    print("Готово! Файлы созданы в LlamaCloud.")


if __name__ == "__main__":
    asyncio.run(main())
