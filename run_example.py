"""
Пример запуска обработки документов.

Использование:
    uv run python run_example.py путь/к/файлу.pdf
    uv run python run_example.py фото1.jpg фото2.jpg фото3.jpg
"""

import asyncio
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from extraction_review.clients import get_llama_cloud_client
from extraction_review.process_file import FileEvent, workflow


async def download_file(client, file_id: str, save_path: Path) -> None:
    """Скачать файл из LlamaCloud."""
    content_info = await client.files.get(file_id)
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(content_info.url)
        response.raise_for_status()
        save_path.write_bytes(response.content)


async def main():
    if len(sys.argv) < 2:
        print("=" * 50)
        print("ОБРАБОТКА ЮРИДИЧЕСКИХ ДОКУМЕНТОВ")
        print("=" * 50)
        print()
        print("Использование:")
        print("  uv run python run_example.py документ.pdf")
        print("  uv run python run_example.py фото1.jpg фото2.jpg фото3.jpg")
        print()
        print("Примеры:")
        print("  uv run python run_example.py иск.pdf")
        print("  uv run python run_example.py стр1.jpg стр2.jpg стр3.jpg")
        return

    files = sys.argv[1:]
    client = get_llama_cloud_client()

    print()
    print("=" * 50)
    print("ОБРАБОТКА ЮРИДИЧЕСКИХ ДОКУМЕНТОВ")
    print("=" * 50)

    print(f"\n📁 Загрузка {len(files)} файл(ов) в облако...")

    # Загружаем файлы в LlamaCloud
    file_ids = []
    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            print(f"❌ Файл не найден: {filepath}")
            return

        print(f"   ✓ {path.name}")
        uploaded = await client.files.create(
            file=str(path),
            purpose="split",
        )
        file_ids.append(uploaded.id)

    # Запускаем обработку
    print("\n🔍 Анализирую структуру документов...")

    if len(file_ids) == 1:
        event = FileEvent(file_id=file_ids[0])
    else:
        event = FileEvent(file_ids=file_ids)

    result = await workflow.run(start_event=event)

    # Выводим результаты
    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("=" * 50)

    if result.merged_from_files > 1:
        print(f"📎 Объединено файлов: {result.merged_from_files}")

    print(f"📄 Всего страниц: {result.total_pages}")
    print(f"📑 Найдено документов: {len(result.segments)}")

    if not result.segments:
        print("\n⚠️ Документы не найдены")
        return

    # Создаём папку для результатов
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print(f"\n📥 Скачиваю файлы в папку '{output_dir}'...")

    for i, seg in enumerate(result.segments, 1):
        save_path = output_dir / seg.filename
        await download_file(client, seg.new_file_id, save_path)

        print(f"\n{i}. ✅ {seg.filename}")
        print(f"   Тип: {seg.category}")
        print(f"   Страницы: {seg.pages}")
        print(f"   Уверенность: {seg.confidence}")

    print("\n" + "=" * 50)
    print(f"✅ ГОТОВО! Файлы сохранены в папку: {output_dir.absolute()}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
