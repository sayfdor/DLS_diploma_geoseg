Сегментация зданий на спутниковых снимках и определение масштаба снимка

Скачать веса модели сегментации и размеченную тестовую выборку: https://drive.google.com/drive/folders/1GMORO7o4L50YMqsqn7dc1LPqYZ7fGRLc?usp=sharing

Архитектура: U-Net с энкодером на основе предобученного ResNet50

Лосс: комбинация Dice Loss, Focal Loss, Boundary Loss

Метрика: IoU

Детекция машин: YOLO26x-obb


Веса модели поместить в папку weights/

Запуск демо-приложения: streamlit run webapp.py

