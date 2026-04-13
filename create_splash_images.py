"""
Генератор заставок в JPG формате для проекта НМУ ВКС.
Создаёт две заставки: предыдущая программа и текущая (ПРОГ2).
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Параметры
W, H = 1200, 700
OUTPUT_FOLDER = 'splash_images'

def create_output_folder():
    """Создаёт папку для заставок."""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def draw_text_on_image(img, text, y_pos, font_size=50, color=(255, 255, 255), bold=False):
    """Вспомогательная функция для добавления текста на изображение."""
    draw = ImageDraw.Draw(img)
    # Пытаемся использовать системный шрифт, если недоступен — используем default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Получаем размер текста
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Центруем по горизонтали
    x_pos = (W - text_width) // 2
    
    # Рисуем текст
    draw.text((x_pos, y_pos), text, font=font, fill=color)

def create_previous_splash():
    """Создаёт заставку предыдущей программы (Система подсчета пассажиров)."""
    # Создаём градиентный фон (синий)
    img = Image.new('RGB', (W, H), color=(20, 40, 80))
    img_array = np.array(img)
    
    # Добавляем градиент
    for y in range(H):
        ratio = y / H
        r = int(20 + (60 - 20) * ratio)
        g = int(40 + (100 - 40) * ratio)
        b = int(80 + (180 - 80) * ratio)
        img_array[y, :] = [r, g, b]
    
    img = Image.fromarray(img_array.astype('uint8'))
    draw = ImageDraw.Draw(img)
    
    # Рисуем верхнюю полоску
    draw.rectangle([(0, 0), (W, 15)], fill=(0, 180, 255))
    draw.rectangle([(0, H - 15), (W, H)], fill=(0, 180, 255))
    
    # Главное название
    draw = ImageDraw.Draw(img)
    try:
        font_main = ImageFont.truetype("arial.ttf", 80)
    except:
        font_main = ImageFont.load_default()
    
    text = "НМУ ВКС"
    bbox = draw.textbbox((0, 0), text, font=font_main)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 100), text, font=font_main, fill=(0, 255, 200))
    
    # Город
    try:
        font_city = ImageFont.truetype("arial.ttf", 40)
    except:
        font_city = ImageFont.load_default()
    
    text = "Воронеж"
    bbox = draw.textbbox((0, 0), text, font=font_city)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 200), text, font=font_city, fill=(180, 180, 180))
    
    # Название программы (предыдущей)
    try:
        font_prog = ImageFont.truetype("arial.ttf", 60)
    except:
        font_prog = ImageFont.load_default()
    
    text = "Система подсчета пассажиров"
    bbox = draw.textbbox((0, 0), text, font=font_prog)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 300), text, font=font_prog, fill=(255, 255, 255))
    
    # Технологии
    try:
        font_tech = ImageFont.truetype("arial.ttf", 28)
    except:
        font_tech = ImageFont.load_default()
    
    text = "OpenCV  |  YOLOv8  |  SQLite  |  Excel"
    bbox = draw.textbbox((0, 0), text, font=font_tech)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 400), text, font=font_tech, fill=(150, 150, 150))
    
    # Контакты
    try:
        font_contact = ImageFont.truetype("arial.ttf", 32)
    except:
        font_contact = ImageFont.load_default()
    
    text = "+7 952 553-96-21  |  mveo@yandex.ru"
    bbox = draw.textbbox((0, 0), text, font=font_contact)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 480), text, font=font_contact, fill=(100, 200, 100))
    
    # Версия
    try:
        font_ver = ImageFont.truetype("arial.ttf", 24)
    except:
        font_ver = ImageFont.load_default()
    
    text = "Программа 1 (День 1-7)"
    bbox = draw.textbbox((0, 0), text, font=font_ver)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 600), text, font=font_ver, fill=(200, 200, 200))
    
    # Сохраняем
    path = os.path.join(OUTPUT_FOLDER, 'splash_previous.jpg')
    img.save(path, quality=95)
    print(f'✓ Создана заставка (предыдущая): {path}')

def create_program2_splash():
    """Создаёт заставку текущей программы (ПРОГ2 - Определение траекторий)."""
    # Создаём градиентный фон (зелёно-синий)
    img = Image.new('RGB', (W, H), color=(10, 50, 30))
    img_array = np.array(img)
    
    # Добавляем градиент
    for y in range(H):
        ratio = y / H
        r = int(10 + (50 - 10) * ratio)
        g = int(50 + (120 - 50) * ratio)
        b = int(30 + (150 - 30) * ratio)
        img_array[y, :] = [r, g, b]
    
    img = Image.fromarray(img_array.astype('uint8'))
    draw = ImageDraw.Draw(img)
    
    # Рисуем верхнюю и нижнюю полоски (яркие)
    draw.rectangle([(0, 0), (W, 15)], fill=(0, 220, 150))
    draw.rectangle([(0, H - 15), (W, H)], fill=(0, 220, 150))
    
    # Главное название
    try:
        font_main = ImageFont.truetype("arial.ttf", 75)
    except:
        font_main = ImageFont.load_default()
    
    text = "НМУ ВКС"
    bbox = draw.textbbox((0, 0), text, font=font_main)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 80), text, font=font_main, fill=(0, 255, 180))
    
    # Город
    try:
        font_city = ImageFont.truetype("arial.ttf", 38)
    except:
        font_city = ImageFont.load_default()
    
    text = "Воронеж"
    bbox = draw.textbbox((0, 0), text, font=font_city)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 170), text, font=font_city, fill=(180, 200, 180))
    
    # *** НОВОЕ НАЗВАНИЕ ПРОГРАММЫ 2 ***
    try:
        font_prog = ImageFont.truetype("arial.ttf", 55)
    except:
        font_prog = ImageFont.load_default()
    
    text = "Определение траекторий"
    bbox = draw.textbbox((0, 0), text, font=font_prog)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 280), text, font=font_prog, fill=(100, 255, 150))
    
    text = "движения пассажиров"
    bbox = draw.textbbox((0, 0), text, font=font_prog)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 360), text, font=font_prog, fill=(100, 255, 150))
    
    # Технологии
    try:
        font_tech = ImageFont.truetype("arial.ttf", 26)
    except:
        font_tech = ImageFont.load_default()
    
    text = "MediaPipe  |  OpenCV  |  YOLOv8  |  SQLite  |  Excel"
    bbox = draw.textbbox((0, 0), text, font=font_tech)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 460), text, font=font_tech, fill=(150, 200, 150))
    
    # Контакты
    try:
        font_contact = ImageFont.truetype("arial.ttf", 30)
    except:
        font_contact = ImageFont.load_default()
    
    text = "+7 952 553-96-21  |  mveo@yandex.ru"
    bbox = draw.textbbox((0, 0), text, font=font_contact)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 535), text, font=font_contact, fill=(100, 220, 100))
    
    # Версия и дата
    try:
        font_ver = ImageFont.truetype("arial.ttf", 23)
    except:
        font_ver = ImageFont.load_default()
    
    text = "Программа 2 (День 1-11)  • Апрель 2026"
    bbox = draw.textbbox((0, 0), text, font=font_ver)
    text_width = bbox[2] - bbox[0]
    x_pos = (W - text_width) // 2
    draw.text((x_pos, 615), text, font=font_ver, fill=(200, 220, 200))
    
    # Сохраняем
    path = os.path.join(OUTPUT_FOLDER, 'splash_program2.jpg')
    img.save(path, quality=95)
    print(f'✓ Создана заставка (ПРОГ2): {path}')

if __name__ == '__main__':
    create_output_folder()
    create_previous_splash()
    create_program2_splash()
    print('\n✓ Обе заставки успешно созданы в папке:', OUTPUT_FOLDER)
