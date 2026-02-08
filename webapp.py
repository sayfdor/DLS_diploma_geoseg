import streamlit as st
import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from src.dataset import InferencePatchDataset
from src.postprocessing import get_vehicle_length_width
from src.model import BuildingSegmentationModel


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_model = BuildingSegmentationModel()
    seg_model.load_state_dict(torch.load("weights/model_weights.pth", map_location=device, weights_only=True))
    seg_model.eval().to(device)

    yolo_model = YOLO("weights/yolo26x-obb.pt")

    return seg_model, yolo_model, device


st.set_page_config(layout="wide")
st.title("Определение площади застройки на спутниковом снимке")

uploaded_file = st.file_uploader("Загрузите скриншот (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    with st.sidebar:
        st.header("Настройки")

        st.subheader("Модель сегментации")
        seg_patch_size = st.slider("Размер патча (сегментация)", 256, 1024, 768, step=64)
        seg_threshold = st.slider("Порог сегментации", 0.1, 0.9, 0.6, step=0.05)
        seg_min_area = st.number_input("Мин. площадь здания (px²)", 10, 5000, 50, step=50)

        st.subheader("Детекция машин")
        det_patch_size = st.slider("Размер патча (детекция)", 256, 1024, 768, step=64)
        det_conf = st.slider("Порог уверенности (машины)", 0.01, 0.5, 0.15, step=0.01)

        st.subheader("Визуализация")
        mask_alpha = st.slider("Прозрачность маски", 0.0, 1.0, 0.6)
        show_cars = st.checkbox("Показывать bbox'ы машин", value=True)

        run_button = st.button("Запустить анализ", type="primary")

    if run_button:
        with st.spinner("Обработка..."):
            seg_model, yolo_model, device = load_models()
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ds_seg = InferencePatchDataset(img_rgb, patch_size=seg_patch_size)
            full_mask = np.zeros((H, W), dtype=np.float32)
            weight_map = np.zeros((H, W), dtype=np.float32)

            for patch, x, y in ds_seg:
                img_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                img_tensor = normalize(img_tensor).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = seg_model(img_tensor).squeeze().cpu().numpy()
                ph, pw = pred.shape
                max_y = min(y + ph, H)
                max_x = min(x + pw, W)
                pred_crop = pred[:max_y - y, :max_x - x]
                full_mask[y:max_y, x:max_x] += pred_crop
                weight_map[y:max_y, x:max_x] += 1

            full_mask = np.where(weight_map > 0, full_mask / weight_map, 0)
            final_mask = (full_mask > seg_threshold).astype(np.uint8)

            ds_det = InferencePatchDataset(img_rgb, patch_size=det_patch_size)
            car_boxes = []
            for patch, x, y in ds_det:
                results = yolo_model(patch, conf=det_conf, verbose=False)
                for r in results:
                    if r.obb is not None:
                        for obb in r.obb:
                            if int(obb.cls.item()) == 10:
                                corners = obb.xyxyxyxy[0].cpu().numpy()
                                corners_global = corners + np.array([x, y])
                                car_boxes.append(corners_global)

            lengths = []
            for corners in car_boxes:
                length, _ = get_vehicle_length_width(corners)
                if 10 < length < 100:
                    lengths.append(length)

            gsd = None
            building_area_m2 = None
            if lengths:
                median_length_px = np.median(lengths)
                AVERAGE_CAR_LENGTH_M = 4.5
                gsd = AVERAGE_CAR_LENGTH_M / median_length_px
                building_area_m2 = final_mask.sum() * (gsd ** 2)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Исходное изображение")
                st.image(img_rgb, use_container_width=True)

            with col2:
                st.subheader("Результат анализа")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(img_rgb)
                ax.imshow(final_mask, cmap='Reds', alpha=mask_alpha)
                if show_cars:
                    for corners in car_boxes:
                        poly = Polygon(corners, closed=True, edgecolor='lime', facecolor='none', linewidth=1)
                        ax.add_patch(poly)
                ax.axis('off')
                st.pyplot(fig, use_container_width=True)

            st.markdown("### Результаты анализа")
            cols = st.columns(4)
            cols[0].metric("Площадь застройки", f"{final_mask.sum():,} px²")
            cols[1].metric("Автомобили", len(car_boxes))
            if gsd is not None:
                cols[2].metric("GSD", f"{gsd:.3f} м/пиксель")
                st.metric("Площадь застройки", f"{building_area_m2:.1f} м²")
            else:
                st.warning("Не удалось оценить GSD: нет автомобилей или они слишком малы.")
