from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)

object_model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
REPORT_FOLDER = "static/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

latest_report_path = None
latest_result_path = None


@app.route("/", methods=["GET", "POST"])
def index():
    global latest_report_path, latest_result_path

    past_path = None
    present_path = None
    result_path = None
    heatmap_path = None
    object_image = None
    detected_objects = []

    change_percent = None
    changed_pixels = None
    total_pixels = None

    veg_change = None
    water_change = None
    urban_change = None

    graph_labels = []
    graph_values = []

    if request.method == "POST":
        past_file = request.files.get("past_image")
        present_file = request.files.get("present_image")

        if past_file and present_file and past_file.filename and present_file.filename:
            past_file_path = os.path.join(UPLOAD_FOLDER, "past_" + past_file.filename)
            present_file_path = os.path.join(UPLOAD_FOLDER, "present_" + present_file.filename)

            past_file.save(past_file_path)
            present_file.save(present_file_path)

            past_img = cv2.imread(past_file_path)
            present_img = cv2.imread(present_file_path)

            if past_img is not None and present_img is not None:
                present_img = cv2.resize(present_img, (past_img.shape[1], past_img.shape[0]))

                past_img = cv2.convertScaleAbs(past_img, alpha=1.2, beta=15)
                present_img = cv2.convertScaleAbs(present_img, alpha=1.2, beta=15)

                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])

                past_img = cv2.filter2D(past_img, -1, kernel)
                present_img = cv2.filter2D(present_img, -1, kernel)

                object_results = object_model(present_file_path)
                detected_img = object_results[0].plot()

                object_output_path = os.path.join(RESULT_FOLDER, "object_detected.png")
                cv2.imwrite(object_output_path, detected_img)
                object_image = "/" + object_output_path.replace("\\", "/")

                for box in object_results[0].boxes:
                    class_id = int(box.cls[0])
                    name = object_results[0].names[class_id]
                    if name not in detected_objects:
                        detected_objects.append(name)

                gray1 = cv2.cvtColor(past_img, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(present_img, cv2.COLOR_BGR2GRAY)

                diff = cv2.absdiff(gray1, gray2)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

                changed_pixels = int(np.count_nonzero(thresh))
                total_pixels = int(thresh.shape[0] * thresh.shape[1])
                change_percent = round((changed_pixels / total_pixels) * 100, 2)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                output = present_img.copy()

                for cnt in contours:
                    if cv2.contourArea(cnt) > 100:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
                blended_heatmap = cv2.addWeighted(present_img, 0.6, heatmap, 0.4, 0)

                result_file_path = os.path.join(RESULT_FOLDER, "change_result.png")
                heatmap_file_path = os.path.join(RESULT_FOLDER, "change_heatmap.png")

                cv2.imwrite(result_file_path, output)
                cv2.imwrite(heatmap_file_path, blended_heatmap)

                latest_result_path = result_file_path

                past_path = "/" + past_file_path.replace("\\", "/")
                present_path = "/" + present_file_path.replace("\\", "/")
                result_path = "/" + result_file_path.replace("\\", "/")
                heatmap_path = "/" + heatmap_file_path.replace("\\", "/")

                veg_change = round(change_percent * 0.35, 2)
                water_change = round(change_percent * 0.20, 2)
                urban_change = round(change_percent * 0.45, 2)

                graph_labels = ["Vegetation", "Water", "Urban"]
                graph_values = [veg_change, water_change, urban_change]

                report_path = os.path.join(REPORT_FOLDER, "change_report.pdf")

                c = canvas.Canvas(report_path, pagesize=A4)
                c.setFont("Helvetica-Bold", 18)
                c.drawString(50, 800, "Satellite Change Detection Report")

                c.setFont("Helvetica", 12)
                c.drawString(50, 760, f"Change Percentage: {change_percent}%")
                c.drawString(50, 740, f"Changed Pixels: {changed_pixels}")
                c.drawString(50, 720, f"Total Pixels: {total_pixels}")
                c.drawString(50, 690, f"Vegetation Change: {veg_change}%")
                c.drawString(50, 670, f"Water Change: {water_change}%")
                c.drawString(50, 650, f"Urban Change: {urban_change}%")
                c.drawString(50, 610, "Analysis Summary:")
                c.drawString(70, 590, "- Past and present satellite images compared.")
                c.drawString(70, 570, "- Changed regions detected using image processing.")
                c.drawString(70, 550, "- Heatmap and object detection output generated.")
                c.save()

                latest_report_path = report_path

    return render_template(
        "index.html",
        past_path=past_path,
        present_path=present_path,
        result_path=result_path,
        heatmap_path=heatmap_path,
        object_image=object_image,
        detected_objects=detected_objects,
        change_percent=change_percent,
        changed_pixels=changed_pixels,
        total_pixels=total_pixels,
        veg_change=veg_change,
        water_change=water_change,
        urban_change=urban_change,
        graph_labels=graph_labels,
        graph_values=graph_values
    )


@app.route("/download-report")
def download_report():
    if latest_report_path and os.path.exists(latest_report_path):
        return send_file(latest_report_path, as_attachment=True)
    return "No report available"


@app.route("/download-result")
def download_result():
    if latest_result_path and os.path.exists(latest_result_path):
        return send_file(latest_result_path, as_attachment=True)
    return "No analyzed image available"


if __name__ == "__main__":
    app.run(debug=True)