from django.shortcuts import render
from .forms import UploadFileForm
from .ml_model import train_and_predict
import pandas as pd
import joblib
import numpy as np

# Upload + Train model
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            df = pd.read_csv(file)
            df.to_csv('predictor/uploaded.csv', index=False)

            accuracy, total = train_and_predict('predictor/uploaded.csv')
            return render(request, 'result.html', {
                'accuracy': round(accuracy * 100, 2),
                'total': total
            })
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})

# Predict new student
def predict_student(request):
    result = None
    if request.method == 'POST':
        attendance = float(request.POST.get('attendance'))
        marks = float(request.POST.get('marks'))
        assignments = float(request.POST.get('assignments'))

        model = joblib.load('predictor/student_model.pkl')
        pred = model.predict([[attendance, marks, assignments]])
        result = 'PASS ✅' if pred[0] == 1 else 'FAIL ❌'

    return render(request, 'predict.html', {'result': result})
