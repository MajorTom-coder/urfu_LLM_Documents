import cv2
import pytesseract
import os
from cv2.gapi import kernel
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
from pathlib import Path

def local_to_absolute_path(file_path):
    return str(Path(file_path).resolve())

# Загрузка модели эмбеддингов
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Сравнение с использованием эмбеддингов напрямую
def compare_names_with_embeddings(threshold=0.8):
    pytesseract.pytesseract.tesseract_cmd = local_to_absolute_path('Tesseract/tesseract.exe')

    approve_arr = [[len(os.listdir(local_to_absolute_path('data/change_image')))],[]]
    agreed_arr = [[len(os.listdir(local_to_absolute_path('data/change_image')))],[]]

    count = 0
    for value in os.listdir(local_to_absolute_path('data/change_image')):
        img = cv2.imread(value)
        data = pytesseract.image_to_data(img, lang='rus', output_type=pytesseract.Output.DICT)

        for iterator in range(len(data)):
            if 'утверждаю' in data['text'][iterator].lower().strip() or 'утверждено' in data['text'][iterator].lower().strip():
                approve_arr[count] = data['text']
            elif 'согласов' in data['text'][iterator].lower().strip():
                agreed_arr[count] = data['text']

        count += 1

    """
    if approve and agreed:
        emb1 = model.encode(approve, convert_to_tensor=True)
        emb2 = model.encode(agreed, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()

        if similarity >= threshold:
            return f"Наименования совпадают (сходство: {similarity:.2f})"
        else:
            return f"Наименования не совпадают (сходство: {similarity:.2f})"
    else:
        return "Ошибка: Не удалось извлечь наименования"
    """

#Преобразование pdf к jpg
def convert_PDF_to_image(file_path, save_image_path = local_to_absolute_path('data/pdf_images'), poppler_path = local_to_absolute_path('urfu_LLM_Documents/lama 3.1/poppler-24.08.0/Library/bin')):
    print('Шаг 1 Конвертация pdf к изображению...')
    delete_file_in_folder()
    os.environ["PATH"] += os.pathsep + poppler_path
    images = convert_from_path(file_path)

    for i, image in enumerate(images, start=1):
        image.save(f'{save_image_path}\\page_{i}.jpg', 'JPEG')

#Удаление предыдущих результатов преобразования pdf к jpg
def delete_file_in_folder(file_path = local_to_absolute_path('data/pdf_images')):
    for filename in os.listdir(file_path):
        file_path = os.path.join(file_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}. {e}')

#Анализ страницы
def get_text_blocks(images_path = local_to_absolute_path('data/pdf_images')):
    print('Шаг 2 Подготовка к обработке изображения')
    pytesseract.pytesseract.tesseract_cmd = local_to_absolute_path('Tesseract/tesseract.exe')

    page_counter = 0

    #Открываем контекст одной страницы
    for i in range(1, len(os.listdir(images_path)) + 1):
        img = cv2.imread(f'{local_to_absolute_path('data/pdf_images')}\\page_{i}.jpg')

        data = pytesseract.image_to_data(img, lang='rus', output_type=pytesseract.Output.DICT)

        boxes = []
        centers = []

        #0 - утверждено, 1 - согласовано
        flag_arr = [False, False]
        left_arr = [0, 0]
        top_arr = [0, 0]

        # Сбор прямоугольников для текста
        for iterator in range(len(data['text'])):

            if 'утверждаю' in data['text'][iterator].lower().strip() or 'утверждено' in data['text'][iterator].lower().strip():
                flag_arr[0] = True
                left_arr[0] = int(data['left'][iterator])
                top_arr[0] = int(data['top'][iterator])

            if 'согласов' in data['text'][iterator].lower().strip():
                flag_arr[1] = True
                left_arr[1] = int(data['left'][iterator])
                top_arr[1] = int(data['top'][iterator])

            if flag_arr[0]:
                if int(data['conf'][iterator]) > 50 and flag_arr[0] and data['text'][iterator] != None and (left_arr[0] - 250 <= data['left'][iterator] or left_arr[0] + 250 >= data['left'][iterator]) and (top_arr[0] - 500 <= data['top'][iterator] and top_arr[0] + 500 >= data['top'][iterator]):
                    x, y, w, h = data['left'][iterator], data['top'][iterator], data['width'][iterator], data['height'][iterator]
                    cx, cy = x + w // 2, y + h // 2
                    boxes.append((x, y, w, h))
                    centers.append([cx, cy])

            if flag_arr[1]:
                if int(data['conf'][iterator]) > 50 and flag_arr[1] and data['text'][iterator] != None and (left_arr[1] - 250 <= data['left'][iterator] or left_arr[1] + 250 >= data['left'][iterator]) and (top_arr[1] - 500 <= data['top'][iterator] and top_arr[1] + 500 >= data['top'][iterator]):
                    x, y, w, h = data['left'][iterator], data['top'][iterator], data['width'][iterator], data['height'][iterator]
                    cx, cy = x + w // 2, y + h // 2
                    boxes.append((x, y, w, h))
                    centers.append([cx, cy])

        if not centers:
            print("Обработка в процессе...")
            continue

        # Кластеризация по координатам центров
        clustering = DBSCAN(eps=160, min_samples=1).fit(centers)
        labels = clustering.labels_

        # Группируем боксы по кластерам
        clusters = {}
        for iterator, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(boxes[iterator])

        cropped_image = img

        # Рисуем рамки вокруг абзацев
        for box_list in clusters.values():
            xs = [x for (x, y, w, h) in box_list]
            ys = [y for (x, y, w, h) in box_list]
            ws = [x + w for (x, y, w, h) in box_list]
            hs = [y + h for (x, y, w, h) in box_list]

            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(ws), max(hs)

            page_counter += 1

            cropped_image = img[y_min - 50:y_max + 50, x_min - 50 :x_max + 50]

            cropped_image_data = pytesseract.image_to_data(cropped_image, lang='rus', output_type=pytesseract.Output.DICT)

            for iterator in range(len(cropped_image_data['text'])):
                if 'утверждаю' in cropped_image_data['text'][iterator].lower().strip() or 'утверждено' in cropped_image_data['text'][iterator].lower().strip() or 'согласов' in cropped_image_data['text'][iterator].lower().strip():
                    cv2.imwrite(f'{local_to_absolute_path('data/change_image')}\\cropped_image_{page_counter}.jpg', cropped_image)
                    print(f'Добавлен файл {local_to_absolute_path('data/change_image')}\\cropped_image_{page_counter}.jpg')




# Основная логика
def main(pdf_path):
    convert_PDF_to_image(pdf_path)
    print(f'Images successfully converted from {pdf_path}')
    get_text_blocks()
    print(f'Обработка завершена!')
    #compare_names_with_embeddings()


# Запуск
pdf_path = 'D:\\OlegDocAnalyze\\fork_urfu_LLM_DOC\\urfu_LLM_Documents\\lama 3.1\\proverka8\\data\\Карталы_Редакция_газеты_оп_4_пх_за_2021_год.pdf'

main(pdf_path)
