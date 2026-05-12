from flask import Flask, render_template, request
import numpy as np
import pickle

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from flask import send_file
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import os


from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prediction.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)




model = pickle.load(open("model/model.pkl", "rb"))
image_model = torch.load("model/image_cancer_model.pth")
latest_prediction = {}

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    radius_mean = db.Column(db.Float)
    texture_mean = db.Column(db.Float)
    perimeter_mean = db.Column(db.Float)
    area_mean = db.Column(db.Float)
    result = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default = datetime.utcnow)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)
        confidence = np.max(probability)*100
        result = ("Malignant" if prediction[0] == 1 else "Benign")

        # save predection to database
        global latest_prediction
        latest_prediction = Prediction(
            radius_mean = features[0],
            texture_mean = features[1],
            perimeter_mean = features[2],
            area_mean = features[3],
            result = result,
            confidence = confidence

        )

        db.session.add(latest_prediction)
        db.session.commit()

        return render_template("index.html", prediction_text = f"prediction: {result} | confidence: {confidence:.2f}%")
    
    except Exception as e:
        return render_template("index.html", prediction_text = f"Error: {str(e)}")
    
@app.route('/history')
def history():
    predictions = Prediction.query.all()
    return render_template("history.html", predictions = predictions)

@app.route('/dashboard')
def dashboard():
    predictions = Prediction.query.all()
    total_predictions = len(predictions)
    benign_count = Prediction.query.filter_by(result = "Benign").count()
    malignant_count = Prediction.query.filter_by(result = "Malignant").count()
    return render_template("dashboard.html", total_predictions = total_predictions, benign_count = benign_count, malignant_count = malignant_count)
@app.route('/download_report')
def download_report():
    global latest_prediction
    file_path ="patient_report.pdf"

    # create pdf
    doc = SimpleDocTemplate(file_path, pagesize = letter)
    styles = getSampleStyleSheet()
    elements=[]
    title = Paragraph("Breast Cancer Prediction Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 20))

    report_line = [
        f"<b>Radius Mean:</b>{latest_prediction.radius_mean}",
        f"<b>Texture Mean:</b>{latest_prediction.texture_mean}",
        f"<b>Perimeter Mean:</b>{latest_prediction.perimeter_mean}",
        f"<b>Area Mean:</b>{latest_prediction.area_mean}",
        f"<b>Prediction Result:</b>{latest_prediction.result}",
        
        f"<b>Confidence:</b>{latest_prediction.confidence}",

    ]
    for line in report_line:
        paragraph = Paragraph(line, styles["BodyText"])
        elements.append(paragraph)
        elements.append(Spacer(1, 12))

    doc.build(elements)
    return send_file(file_path, as_attachment=True)

class BreastCancerCNN(nn.Module):

    def __init__(self):
        super(BreastCancerCNN, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, 2)
        )

    def forward(self, x):

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_model = BreastCancerCNN().to(DEVICE)

image_model.load_state_dict(
    torch.load(
        "model/image_cancer_model.pth",
        map_location=DEVICE
    )
    )

image_model.eval()
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
    ])

@app.route('/image_predict', methods= ['GET', 'POST'])
def image_predict():
    if request.method == 'GET':
        return render_template('image_predict.html')
    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return render_template('image_predict.html', Prediction = 'noimage upload')
        image = Image.open(file).convert('RGB')
        image = transform(image)
        image =image.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
                outputs = image_model(image)
                _,predicted = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)
                conifidence = torch.max(probabilities).item()*100
        result = "Malignant" if predicted.item() == 1 else "Benign"
        return render_template("result.html", result = result, conifidence = f"{conifidence:.2f}")
    return render_template('image_predict.html')



if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
